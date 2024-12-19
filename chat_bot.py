import re
from revChatGPT.V3 import Chatbot
from transformers import pipeline
import time
from text2vec import SentenceModel
from googlesearch import search
from cxr.prompt import prob2text
from r2g.report_generate import reportGen
from cxr.diagnosis import getJFImg, JFinfer, JFinit
import json
from engine_LLM.api import answer_quest, query_range
from modality_identify import ModalityClip


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, list) and all(isinstance(x, int) for x in obj):
            return obj
        return super().default(obj)


fivedisease = {
    "Cardiomegaly": 0,
    "Edema": 1,
    "Consolidation": 2,
    "Atelectasis": 3,
    "Pleural Effusion": 4,
}

chest_base_dict = {
    "Atelectasis": 0,
    "Pleural Effusion": 2,
    "Mediastinal Mass": 4,
    "Aspiration Pneumonia": 6.1,
    "Community-Acquired Pneumonia": 6.2,
    "Hospital-Acquired Pneumonia": 6.3,
    "Ventilator-Associated Pneumonia": 6.4,
    "Immunocompromised Pneumonia": 6.5,
    "Pneumothorax": 7,
    "Pulmonary Edema": 9,
    "Chronic Obstructive Pulmonary Disease (COPD)": 10,
    "Pleural Fibrosis and Calcification": 11,
    "Diaphragmatic Hernia": 13,
}


class base_bot:
    def start(self):
        """Initialize the chatbot for the current session."""
        pass

    def reset(self):
        """Reset the current chatbot session."""
        pass

    def chat(self, message: str):
        pass


class gpt_bot(base_bot):
    def __init__(self, engine: str, api_key: str):
        """Initialize the model."""
        self.agent = None
        self.engine = engine
        self.api_key = api_key
        img_model, imgcfg = JFinit("./cxr/config/JF.json", "./weights/JFchexpert.pth")
        self.imgcfg = imgcfg
        self.img_model = img_model
        self.reporter = reportGen()
        self.modality = ["chest x-ray", "panoramic dental x-ray", "knee MRI", "Mammography"]
        self.translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en", device=0)
        self.identifier = ModalityClip(self.modality)
        self.sent_model = SentenceModel()
        self.msd_dict = json.load(open("./engine_LLM/dataset/msd_dict.json", "r", encoding="utf-8"))

    def ret_local(self, query: str, mode=1):
        topic_range = query_range(self.sent_model, query, k=1, bar=0.0)
        if mode == 0:  # Chinese
            return "https://" + self.msd_dict[topic_range[0]]
        else:  # English
            return "https://" + self.msd_dict[topic_range[0]].replace("www.msdmanuals.cn", "www.merckmanuals.com")

    def translate_zh_to_en(self, content: str):
        """Translate Chinese text to English."""
        output = self.translator(content)
        report_en = output[0]["translation_text"]
        return report_en

    def chat_with_gpt(self, prompt):
        iteration = 0
        while True:
            iteration += 1
            print(f"talking {iteration}......")
            try:
                message = self.agent.ask(prompt)
            except:
                time.sleep(10)
                continue
            break
        return message

    def start(self):
        """Start a new chatbot session."""
        if self.agent is not None:
            self.agent.reset()
        system_prompt = "You are ChatCAD-plus, a universal and reliable CAD system. Respond conversationally."
        self.agent = Chatbot(engine=self.engine, api_key=self.api_key, system_prompt=system_prompt, proxy="http://127.0.0.1:7890")
        instruction = "Act as a doctor named ChatCAD-plus. Unless specified, all your answers should be in English."
        res = self.chat_with_gpt(instruction)
        print(res)
        return

    def report_cxr_en(self, img_path, mode: str = "run"):
        img1, img2 = getJFImg(img_path, self.imgcfg)
        text_report = self.reporter.report(img1)[0]
        text_report = self.translate_zh_to_en(text_report)
        prob = JFinfer(self.img_model, img2, self.imgcfg)
        converter = prob2text(prob, fivedisease)
        res = converter.promptB()
        prompt_report = f"Diagnosis report: {res}. Report details: {text_report}."
        print("Refine CXR reports...")
        refined_report = self.chat_with_gpt(prompt_report)
        if mode == "debug":
            return text_report, refined_report, prob.detach().cpu().numpy().tolist()
        else:
            return refined_report

    def chat(self, message: str, ref_record: str):
        """Chat with the user."""
        response = self.chat_with_gpt(f"{ref_record}\nUser: {message}\nResponse:")
        return response
