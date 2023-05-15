import re
import unidecode
import spacy

nlp = spacy.load('en_core_web_sm')


def preprocessing(resume_txt: str) -> str:
    new_resume_txt = re.sub(r'<.*?>', '', resume_txt)
    new_resume_txt = re.sub(r'\s+', ' ', new_resume_txt)
    new_resume_txt = new_resume_txt.replace('\n', ' ')
    new_resume_txt = unidecode.unidecode(new_resume_txt)
    doc = nlp(new_resume_txt)
    tokens = [t.lemma_.lower() for t in doc if t.pos_ != 'PUNCT' and not t.is_stop]
    tokens = [t.text.lower() for t in doc if t.pos_ != 'PUNCT']
    new_resume_txt = ' '.join(tokens).strip()
    return new_resume_txt.lower()
