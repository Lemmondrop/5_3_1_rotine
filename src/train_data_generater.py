import json
import random

# 2.5kg 단위 반올림 함수
def round_to_nearest_2_5(x):
    return round(x / 2.5) * 2.5

# 훈련 1RM 계산 함수
def calculate_training_max(one_rm):
    raw_tm = one_rm * 0.9
    rounded_tm = round_to_nearest_2_5(raw_tm)
    return raw_tm, rounded_tm

# 주차별 설정
def get_week_configs():
    return {
        1: {'reps': ['5', '5', '5+'], 'percentages': [0.65, 0.75, 0.85]},
        2: {'reps': ['3', '3', '3+'], 'percentages': [0.70, 0.80, 0.90]},
        3: {'reps': ['5', '3', '1+'], 'percentages': [0.75, 0.85, 0.95]},
        4: {'reps': ['5', '5', '5'], 'percentages': [0.40, 0.50, 0.60]},
    }

# 세트별 계산 로직 (CoT 포함)
def generate_week_weights_with_cot(training_max_rounded, week_num):
    config = get_week_configs()[week_num]
    sets_output = []
    for p, r in zip(config['percentages'], config['reps']):
        raw = training_max_rounded * p
        rounded = round_to_nearest_2_5(raw)
        cot_str = f"{training_max_rounded:.1f}kg * {int(p*100)}% = {raw:.1f}kg -> {rounded:.1f}kg x {r}"
        sets_output.append(cot_str)
    return sets_output

# 전체 루틴 생성
def create_full_routine(days, level, goal, one_rms):
    output = []
    training_maxes = {}

    output.append("")
    for ex, rm in one_rms.items():
        raw_tm, rounded_tm = calculate_training_max(rm)
        training_maxes[ex] = rounded_tm
        output.append(f"- {ex} 1RM: {rm}kg")
        output.append(f"  -> 훈련 1RM (90%): {rm} * 0.9 = {raw_tm:.1f}kg -> {rounded_tm:.1f}kg (2.5kg 단위 반올림)")
    output.append("")

    for week in range(1, 5):
        output.append(f"{week}주차 루틴 ({days}일 루틴)\n")
        s = generate_week_weights_with_cot(training_maxes['스쿼트'], week)
        b = generate_week_weights_with_cot(training_maxes['벤치프레스'], week)
        d = generate_week_weights_with_cot(training_maxes['데드리프트'], week)
        o = generate_week_weights_with_cot(training_maxes['오버헤드 프레스'], week)

        if days == 3:
            output.extend([
                "월 (스쿼트 + 벤치프레스)", f"- 스쿼트: {', '.join(s)}", f"- 벤치프레스: {', '.join(b)}",
                "수 (데드리프트 + 오버헤드 프레스)", f"- 데드리프트: {', '.join(d)}", f"- 오버헤드 프레스: {', '.join(o)}",
                "금 (스쿼트 + 벤치프레스)", f"- 스쿼트: {', '.join(s)}", f"- 벤치프레스: {', '.join(b)}",
            ])
        else:
            output.extend([
                "월 (스쿼트)", f"- 스쿼트: {', '.join(s)}",
                "화 (벤치프레스)", f"- 벤치프레스: {', '.join(b)}",
                "수 (데드리프트)", f"- 데드리프트: {', '.join(d)}",
                "목 (오버헤드 프레스)", f"- 오버헤드 프레스: {', '.join(o)}",
                "금 (스쿼트 + 벤치프레스)", f"- 스쿼트: {', '.join(s)}", f"- 벤치프레스: {', '.join(b)}",
            ])

        if week < 4:
            output.append("")

    return "\n".join(output).strip()

# 설정 변수들
levels = ["초보자", "중급자", "숙련자"]
goals = ["체중 증가", "근력 향상", "체지방 감소"]
instructions = [
    "다음 사용자 정보를 바탕으로 5/3/1 운동 루틴을 생성하세요. 모든 계산 과정을 포함하고 정확한 5/3/1 규칙을 따르세요.",
    "사용자의 정보를 참고하여 적절한 5/3/1 루틴을 설계하세요. 모든 훈련 1RM과 세트별 중량 계산 과정을 명시적으로 보여주세요.",
    "아래 정보를 참고해 한국어로 5/3/1 루틴을 작성하세요. 훈련 중량 및 세트별 중량 계산 과정을 상세히 포함합니다.",
    "5/3/1 규칙에 따라 사용자 맞춤 루틴을 구성하세요. 모든 단계별 계산을 보여주세요."
]

def generate_sample():
    days = random.choice([3, 5])
    level = random.choice(levels)
    goal = random.choice(goals)
    instruction = random.choice(instructions)

    def random_rm(low, high):
        return round_to_nearest_2_5(random.randint(low, high))

    one_rms = {
        "스쿼트": random_rm(60, 160),
        "벤치프레스": random_rm(40, 120),
        "데드리프트": random_rm(80, 200),
        "오버헤드 프레스": random_rm(25, 70),
    }

    input_text = f"운동 가능일: {days}일\n운동 경험: {level}\n운동 목표: {goal}\n1RM 정보:\n" + \
        f"- 스쿼트: {one_rms['스쿼트']}kg\n" + \
        f"- 벤치프레스: {one_rms['벤치프레스']}kg\n" + \
        f"- 데드리프트: {one_rms['데드리프트']}kg\n" + \
        f"- 오버헤드 프레스: {one_rms['오버헤드 프레스']}kg"

    output = create_full_routine(days, level, goal, one_rms)

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output
    }

# 샘플 생성 및 저장
data = [generate_sample() for _ in range(5000)]
with open("generated_531_routines_full_cot_5000.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"총 {len(data)}개의 5/3/1 루틴 샘플이 'generated_531_routines_full_cot_5000.json' 파일로 생성되었습니다.")