import torch
import time
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# [0] 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 사용 디바이스: {device}")
if device.type == "cpu":
    print("⚠️ 경고: 현재 GPU를 사용할 수 없습니다. float16 모델은 CPU에서 매우 느리게 작동합니다.")

# [1] 모델 설정
base_model = "kakaocorp/kanana-nano-2.1b-instruct"
lora_path = "./results_kanana_v2"

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(
    base_model, trust_remote_code=True, torch_dtype=torch.float16
).to(device)

model = PeftModel.from_pretrained(base, lora_path).to(device).eval()

# [2] 프롬프트 생성 함수
def build_prompt(user_info: str, week: int) -> str:
    return f"""
[사용자 정보]
{user_info}

[요구사항]
당신은 퍼스널 트레이너입니다. 아래 정보를 바탕으로 5/3/1 웨이트 트레이닝 루틴 중 **[{week}주차 루틴]** 만 생성하세요.

운동은 요일별로 구성됩니다:
- [월요일]: 벤치프레스
- [화요일]: 스쿼트
- [수요일]: 휴식
- [목요일]: 오버헤드프레스
- [금요일]: 데드리프트

훈련 1RM은 입력된 1RM의 90%입니다. 각 운동은 요일별로 3세트 구성하며, 중량은 2.5kg 단위로 반올림하세요. 각 세트에 대해 계산 과정, 반올림 결과, 반복 횟수를 포함하세요.

출력 형식 예시:
[1주차 루틴]
[월요일]
- 벤치프레스
  - 1세트: 80kg × 0.9 × 0.65 = 46.8kg → 47.5kg × 5회
  - 2세트: 80kg × 0.9 × 0.75 = 54.0kg → 55.0kg × 5회
  - 3세트: 80kg × 0.9 × 0.85 = 61.2kg → 60.0kg × AMRAP
"""

# [3] 1RM 추출 함수
def extract_one_rm(user_info: str) -> dict:
    pattern = r"(스쿼트|벤치프레스|데드리프트|오버헤드 프레스)[\s:]?(\d+(?:\.\d+)?)kg"
    matches = re.findall(pattern, user_info)
    return {exercise: float(weight) for exercise, weight in matches}

# [4] 후처리 함수
def postprocess(text: str, one_rm: dict, week: int) -> str:
    week_plan = {
        1: {"percentages": [0.65, 0.75, 0.85], "reps": ["5회", "5회", "AMRAP"]},
        2: {"percentages": [0.70, 0.80, 0.90], "reps": ["3회", "3회", "AMRAP"]},
        3: {"percentages": [0.75, 0.85, 0.95], "reps": ["5회", "3회", "AMRAP"]},
        4: {"percentages": [0.40, 0.50, 0.60], "reps": ["5회", "5회", "5회"]},
    }

    plan = week_plan.get(week, week_plan[1])

    # 마크다운 제거 및 문구 정리
    text = re.sub(r"```markdown[\s\S]+?```", "", text)
    text = re.sub(r"(운동 전.*?전문가에게 문의하세요\.)+", "운동 전 충분한 준비운동과 마무리운동을 잊지 마세요.", text)

    # 기존 반복 횟수 제거 (중복 방지)
    text = re.sub(r"(→ [\d\.]+kg)( × (AMRAP|\d+회|\d+\+회))+", r"\1", text)

    # 수요일 휴식 삽입
    def insert_wednesday_rest(block):
        if "[수요일]" not in block:
            return re.sub(r"(\[\d주차 루틴\][\s\S]*?)(\[\w{3,}요일\])", r"\1[수요일]\n- 휴식\n\n\2", block, count=1)
        return block

    text = re.sub(r"(\[\d주차 루틴\][\s\S]*?)(?=\[\d주차 루틴|\Z)", lambda m: insert_wednesday_rest(m.group(0)), text)

    # 금요일 3세트 보완 삽입
    def fix_friday_last_set(block):
        if "[금요일]" in block:
            lines = block.splitlines()
            idx = next((i for i, line in enumerate(lines) if "[금요일]" in line), -1)
            if idx != -1:
                lines = [line for i, line in enumerate(lines) if not (i > idx and "3세트" in line)]
                try:
                    rm = one_rm.get("데드리프트", 140.0)
                    third_raw = round(rm * 0.9 * plan["percentages"][2], 1)
                    third_rounded = round(round(third_raw / 2.5) * 2.5, 1)
                    third_rounded = int(third_rounded) if third_rounded.is_integer() else third_rounded
                    formula = f"{rm:.1f}kg × 0.9 × {plan['percentages'][2]}"
                    new_line = (
                        f"  - 3세트 ({int(plan['percentages'][2]*100)}% × {plan['reps'][2]}): "
                        f"{formula} = {third_raw}kg → {third_rounded}kg"
                    )
                    insert_at = idx + 4 if len(lines) > idx + 3 else len(lines)
                    lines.insert(insert_at, new_line)
                except Exception:
                    pass
            return "\n".join(lines)
        return block

    text = re.sub(r"(\[\d주차 루틴\][\s\S]*?)(?=\[\d주차 루틴|\Z)", lambda m: fix_friday_last_set(m.group(0)), text)

    # 강제 수식 삽입 (전체 라인 대상)
    def apply_forced_weight(block):
        current_ex = None
        set_idx = 0
        result_lines = []
        for line in block.splitlines():
            ex_match = re.match(r"- ([가-힣\s]+)", line)
            if ex_match:
                current_ex = ex_match.group(1).strip()
                set_idx = 0
                result_lines.append(line)
                continue
            set_match = re.match(r"\s+- (\d)세트", line)
            if set_match and current_ex:
                idx = int(set_match.group(1)) - 1
                rm = one_rm.get(current_ex, 100.0)
                tm = rm * 0.9
                perc = plan["percentages"][idx]
                reps = plan["reps"][idx]
                raw = round(tm * perc, 1)
                rounded = round(round(raw / 2.5) * 2.5, 1)
                rounded = int(rounded) if rounded.is_integer() else rounded
                formula = f"{rm:.1f}kg × 0.9 × {perc}"
                fixed = f"  - {idx+1}세트 ({int(perc*100)}% × {reps}): {formula} = {raw}kg → {rounded}kg"
                result_lines.append(fixed)
            else:
                result_lines.append(line)
        return "\n".join(result_lines)

    text = re.sub(r"(\[\d주차 루틴\][\s\S]*?)(?=\[\d주차 루틴|\Z)", lambda m: apply_forced_weight(m.group(0)), text)

    # 첫 주차부터 시작되도록 정리
    first_week = re.search(r"\[\d주차 루틴\]", text)
    if first_week:
        text = text[first_week.start():]

    # 결과 또는 참고 이후 텍스트 제거
    text = re.split(r"\[결과\]|\[참고\]", text)[0]

    # 중복된 주차 루틴 제거
    lines = text.splitlines()
    seen = set()
    result = []
    for line in lines:
        if re.match(r"\[\d주차 루틴\]", line):
            if line in seen:
                continue
            seen.add(line)
        result.append(line)

    return "\n".join(result).strip()

# [5] 루틴 생성 함수
def generate_routine():
    user_info = (
        "운동 경험: 고급자\n"
        "운동 목표: 근력 유지\n"
        "1RM 정보: 스쿼트 180kg, 벤치프레스 120kg, 데드리프트 200kg, 오버헤드 프레스 80kg"
    )

    one_rm = extract_one_rm(user_info)

    all_weeks = []
    total_start = time.time()

    for week in range(1, 5):
        print(f"\n--- {week}주차 루틴 생성 시작 ---")
        t0 = time.time()

        prompt = build_prompt(user_info, week)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs['input_ids'].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=865,
                temperature=0.7,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        t1 = time.time()
        print(f"[{week}주차] generate() 소요 시간: {t1 - t0:.2f}초")

        output_ids = outputs[0][input_len:]
        decoded = tokenizer.decode(output_ids, skip_special_tokens=True)

        t2 = time.time()
        cleaned = postprocess(decoded, one_rm, week)
        t3 = time.time()
        print(f"[{week}주차] 후처리 소요 시간: {t3 - t2:.2f}초")

        all_weeks.append(cleaned)

    total_end = time.time()
    print(f"\n✅ 전체 소요 시간: {total_end - total_start:.2f}초")

    full_routine = "\n\n".join(all_weeks)
    print("\n📌 생성된 루틴:\n")
    print(full_routine)

# [6] 실행
if __name__ == "__main__":
    generate_routine()