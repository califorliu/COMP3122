from core_system import NoteSummarizerAI
import os


def main():
    ai = NoteSummarizerAI()

    # 1. P端：处理 upload_file 目录下的所有课件
    files = [f for f in os.listdir("./upload_file") if os.path.isfile(os.path.join("./upload_file", f))]
    for f in files:
        print(f"正在处理课件: {f}")
        count = ai.extract_from_file(f, domain="计算机科学")
        if count:
            ai.verify_course_terms(f)
            print(f"  - 成功提取并验证 {count} 个术语。")

    # 2. S端：学生提问
    print("\n--- 学生提问环节 ---")
    res = ai.student_qa("S001", "请解释一下什么是过拟合？")
    print(f"AI回答: {res.get('answer', {}).get('response', {}).get('direct_answer', '无回答')}")


if __name__ == "__main__":
    main()