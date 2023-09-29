import torch
import torchaudio
from seamless_communication.models.inference import Translator


# translator = Translator(
#     "seamlessM4T_medium",
#     "vocoder_36langs",
#     torch.device("cuda:0"),
#     torch.float16
# )



class ContentToTransform():

    lang_code = {"English":"eng","Filipino":"tgl","Korean":"kor"}


    def __init__ (self,content,language):
        self.content = content
        self.language = language

    def to_translate(self, model: Translator):
        translated_text, _, _ = model.predict(self.content, "t2tt", self.lang_code[f"{self.language}"], src_lang=self.lang_code["English"])

        return {"translated_content":str(translated_text)}

    def to_speech(self, model: Translator):
        translated_text, wav, sr = model.predict(self.content, "t2st", self.lang_code[f"{self.language}"], src_lang=self.lang_code["English"])

        torchaudio.save(
            "test.wav",
            wav[0].cpu(),
            sample_rate=sr,
        )

        return "test.wav"

# test = ContentToTransform(content="Iam joe nice to meet you",language="Filipino")
# print(test.to_translate(model=translator))