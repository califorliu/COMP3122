import time
import uuid
import os
from config import UPLOAD_DIR
from database import LocalDB
from llm_client import call_moonshot_json
from file_processor import FileProcessor


class NoteSummarizerAI:
    # 阶段 1：从特定课件提取术语
    def extract_from_file(self, filename, domain="通用"):
        file_path = os.path.join(UPLOAD_DIR, filename)
        content = FileProcessor.process_file(file_path)
        if not content: return

        system_prompt = f"你是一个术语提取专家。请从文档中提取专业术语。学科：{domain}。必须返回JSON格式。"
        user_prompt = f"文档内容：\n{content[:4000]}"  # 截断防止超长

        result = call_moonshot_json(system_prompt, user_prompt)
        if result and "cached_terms" in result:
            terms_db = {}
            for term in result["cached_terms"]:
                t_id = f"term_{uuid.uuid4().hex[:6]}"
                term.update({"term_id": t_id, "source_file": filename, "status": "pending_verification"})
                terms_db[t_id] = term
            LocalDB.save_course_terms(filename, terms_db)
            return len(terms_db)
        return 0

    # 阶段 2：验证特定课件的术语
    def verify_course_terms(self, filename):
        terms = LocalDB.get_course_terms(filename)
        if not terms: return

        system_prompt = "你是一个术语验证专家。请根据通用知识与课件定义进行比对，修正解释。返回JSON。"
        user_prompt = f"待验证数据：{list(terms.values())}"

        result = call_moonshot_json(system_prompt, user_prompt)
        if result and "verified_terms" in result:
            for vt in result["verified_terms"]:
                for tid, tdata in terms.items():
                    if tdata["term"] == vt["term"]:
                        tdata.update(vt)
                        tdata.update({"status": "verified", "verified_at": time.ctime()})
            LocalDB.save_course_terms(filename, terms)

    # 阶段 3：学生答疑
    def student_qa(self, student_id, question):
        verified_cache = LocalDB.get_all_verified_terms()
        matched = [t for t in verified_cache.values() if t["term"].lower() in question.lower()]

        if not matched:
            return {"response": {"direct_answer": "该概念不在当前课件范围。"}}

        system_prompt = "你是一个答疑助手。必须基于提供的缓存术语回答，并引用课件来源。"
        user_prompt = f"问题：{question}\n缓存数据：{matched}"

        answer = call_moonshot_json(system_prompt, user_prompt)
        if answer:
            session = {"session_id": uuid.uuid4().hex, "question": question, "answer": answer,
                       "timestamp": time.ctime()}
            LocalDB.save_student_session(student_id, session)
            return session
        return {"error": "无法生成回答"}