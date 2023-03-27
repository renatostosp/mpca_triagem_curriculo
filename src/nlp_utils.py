import re
import unidecode


def preprocessing(resume_txt: str) -> str:
    new_resume_txt = re.sub(r'<.*?>', '', resume_txt)  # remove extra whitespace
    new_resume_txt = re.sub(r'\s+', ' ', new_resume_txt)  # remove extra whitespace
    new_resume_txt = new_resume_txt.replace('\n', ' ')
    new_resume_txt = unidecode.unidecode(new_resume_txt)
    return new_resume_txt.lower()
