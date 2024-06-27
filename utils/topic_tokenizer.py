import sudachipy
import re

class Tokenizer:
    def __init__(self,lang:str):
        self.stop_word = self.get_stop_words(lang)
        self.lang = lang
        self.en = re.compile(r'[a-zA-Z]+')
        self.jp = re.compile(r'[ぁ-んァ-ン一-龥]+')
        self.jpsplit = re.compile(r'[。\n！？]')
        self.sudachi_tokenizer = sudachipy.Dictionary().create(mode=sudachipy.SplitMode.C)
    def __call__(self, text:str):
        if self.lang == "jp":
            words = self.sudachi(text)
            return [w for w in words if w not in self.stop_word and self.jp.fullmatch(w)]
        elif self.lang == "en":
            return [w.lower() for w in text.split() if w.lower() not in self.stop_word and self.en.fullmatch(w)]
        else:
            raise ValueError("lang must be 'jp' or 'en'")
    def sudachi(self,text):
        s=text
        if len(s)>15000:
            # print(f"too long sentence of {len(s)}:",s)
            s=s[:15000]
        ms = self.sudachi_tokenizer.tokenize(s)
        words = [m.surface() for m in ms]
        return words
    def sudachi_(self,text):
        words = []
        for s in self.jpsplit.split(text):
            # for s in t.split("。"):
            if len(s)>10000:
                print(f"too long sentence of {len(s)}:",s)
                s=s[:10000]
            ms = self.sudachi_tokenizer.tokenize(s)
            words += [m.surface() for m in ms]
        return words
    def sudachi_wakachi(self,text):
        words = []
        wakati_text = []
        for t in text.split("\n"):
            for s in t.split("。"):
                if len(s)>10000:   # this is extremely rare for Japanese sentences
                    print(f"too long sentence of {len(s)}:",s)
                    s=s[:10000]
                ms = self.sudachi_tokenizer.tokenize(s)
                for m in ms:
                    word = m.surface()
                    wakati_text.append(word)
                    if m.part_of_speech()[0] == "名詞":
                        if num.fullmatch(word):
                            words.append("0")
                        elif symbol.fullmatch(word) or alpha.fullmatch(word):
                            pass
                        else:
                            words.append(word)
        wakati_text = " ".join(wakati_text)
        return words, wakati_text

    def get_stop_words(self,lang):
    # if lang=="jp":
    #     url = "http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt"
    # elif lang=="en":
    #     url = "https://gist.githubusercontent.com/sebleier/554280/raw/7e0e4a1ce04c2bb7bd41089c9821dbcf6d0c786c/NLTK's%2520list%2520of%2520english%2520stopwords"
    # else:
    #     raise
    # r = requests.get(url)
    # tmp = r.text.split('\r\n')
        stopwords = open(f"stopwords/{lang}.txt", "r").readlines()
        stopwords = [s.strip() for s in stopwords]
        if lang=="jp":
            # stopwords = []
            # for i in range(len(tmp)):
            #     if len(tmp[i]) < 1:
            #         continue
            #     stopwords.append(tmp[i])
            stopwords += ["０","１","２","３","４","５","６","７","８","９"]
        # else:
        #     stopwords = tmp[0].split('\n')
        return stopwords

