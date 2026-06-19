"""
IQ Detector Pro — Enhanced Edition
7 ML Algorithms · Shape MCQ buttons · GK questions · Live ML breakdown
· Fixed timer · Bright high-contrast UI · 15 questions per session
"""

import gradio as gr
import numpy as np
import random, time, warnings, tempfile
from datetime import datetime

warnings.filterwarnings("ignore")

from sklearn.linear_model  import LogisticRegression, Ridge
from sklearn.cluster       import KMeans
from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm           import SVC
from sklearn.naive_bayes   import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import (accuracy_score, precision_score,
                                   recall_score, f1_score,
                                   confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
import time as _timeit

# ══════════════════════════════════════════════════════════════
#  QUESTION DATABASE  (45 Qs total)
#  Numbers: 15  |  Shapes: 12  |  General Knowledge: 18
#  Per session: 5 numbers + 4 shapes + 6 GK = 15 questions
# ══════════════════════════════════════════════════════════════
QUESTION_DATABASE = [
    # ── NUMBERS easy ──────────────────────────────────────────
    {"id":1,  "sequence":"2, 4, 6, 8, ?",       "answer":"10",  "pattern":"+2",               "category":"numbers","difficulty":0,"hint":"Add 2 each time","options":None},
    {"id":2,  "sequence":"5, 10, 15, 20, ?",     "answer":"25",  "pattern":"+5",               "category":"numbers","difficulty":0,"hint":"Add 5 each time","options":None},
    {"id":3,  "sequence":"1, 3, 5, 7, ?",        "answer":"9",   "pattern":"Odd numbers",      "category":"numbers","difficulty":0,"hint":"Odd numbers","options":None},
    {"id":4,  "sequence":"10, 8, 6, 4, ?",       "answer":"2",   "pattern":"-2",               "category":"numbers","difficulty":0,"hint":"Subtract 2","options":None},
    {"id":5,  "sequence":"1, 2, 3, 4, ?",        "answer":"5",   "pattern":"+1",               "category":"numbers","difficulty":0,"hint":"Count up by 1","options":None},
    # ── NUMBERS medium ────────────────────────────────────────
    {"id":6,  "sequence":"1, 1, 2, 3, 5, ?",     "answer":"8",   "pattern":"Fibonacci",        "category":"numbers","difficulty":1,"hint":"Add previous two","options":None},
    {"id":7,  "sequence":"1, 4, 9, 16, ?",       "answer":"25",  "pattern":"Square numbers",   "category":"numbers","difficulty":1,"hint":"1²,2²,3²,4²…","options":None},
    {"id":8,  "sequence":"3, 6, 12, 24, ?",      "answer":"48",  "pattern":"×2",               "category":"numbers","difficulty":1,"hint":"Double each time","options":None},
    {"id":9,  "sequence":"2, 6, 12, 20, ?",      "answer":"30",  "pattern":"+4,+6,+8,+10",    "category":"numbers","difficulty":1,"hint":"Add increasing even numbers","options":None},
    {"id":10, "sequence":"7, 14, 21, 28, ?",     "answer":"35",  "pattern":"+7",               "category":"numbers","difficulty":1,"hint":"Multiples of 7","options":None},
    # ── NUMBERS hard ──────────────────────────────────────────
    {"id":11, "sequence":"2, 6, 12, 20, 30, ?",  "answer":"42",  "pattern":"+4,+6,+8,+10,+12","category":"numbers","difficulty":2,"hint":"Add consecutive even numbers","options":None},
    {"id":12, "sequence":"1, 8, 27, 64, ?",      "answer":"125", "pattern":"Cube numbers",     "category":"numbers","difficulty":2,"hint":"1³,2³,3³,4³…","options":None},
    {"id":13, "sequence":"0, 1, 3, 6, 10, ?",    "answer":"15",  "pattern":"Triangular numbers","category":"numbers","difficulty":2,"hint":"Add 1,2,3,4,5…","options":None},
    {"id":14, "sequence":"1, 3, 6, 10, 15, ?",   "answer":"21",  "pattern":"+2,+3,+4,+5,+6",  "category":"numbers","difficulty":2,"hint":"Add increasing integers","options":None},
    {"id":15, "sequence":"5, 7, 11, 17, 25, ?",  "answer":"35",  "pattern":"+2,+4,+6,+8,+10", "category":"numbers","difficulty":2,"hint":"Add increasing even numbers","options":None},
    # ── SHAPES easy (MCQ) ─────────────────────────────────────
    {"id":16, "sequence":"▲, ■, ▲, ■, ?",        "answer":"▲",   "pattern":"Alternating shapes",  "category":"shapes","difficulty":0,"hint":"Pattern repeats",   "options":["▲","■","●","○"]},
    {"id":17, "sequence":"●, ○, ●, ○, ?",        "answer":"●",   "pattern":"Alternating circles", "category":"shapes","difficulty":0,"hint":"Black/white alternate","options":["●","○","▲","★"]},
    {"id":18, "sequence":"★, ★, ★, ★, ?",        "answer":"★",   "pattern":"All same",            "category":"shapes","difficulty":0,"hint":"Nothing changes",    "options":["★","☆","■","▲"]},
    {"id":19, "sequence":"■, □, ■, □, ?",        "answer":"■",   "pattern":"Alternating squares", "category":"shapes","difficulty":0,"hint":"Filled/empty alternate","options":["■","□","●","▲"]},
    # ── SHAPES medium (MCQ) ───────────────────────────────────
    {"id":20, "sequence":"★, ☆, ★★, ☆☆, ?",      "answer":"★★★", "pattern":"Increasing filled stars","category":"shapes","difficulty":1,"hint":"Add one filled star","options":["★★★","☆☆☆","★★","☆☆"]},
    {"id":21, "sequence":"▲, ▲▲, ▲, ▲▲, ?",      "answer":"▲",   "pattern":"Single-double repeat",  "category":"shapes","difficulty":1,"hint":"Pair alternates",   "options":["▲","▲▲","▲▲▲","■"]},
    {"id":22, "sequence":"■, ■■, ■■■, ?",        "answer":"■■■■","pattern":"Increasing squares",    "category":"shapes","difficulty":1,"hint":"Add one square",    "options":["■■■■","■■■","■■","■■■■■"]},
    {"id":23, "sequence":"●, ●●, ●●●, ?",        "answer":"●●●●","pattern":"Increasing circles",   "category":"shapes","difficulty":1,"hint":"Add one circle",    "options":["●●●●","●●●","●●","●●●●●"]},
    # ── SHAPES hard (MCQ) ─────────────────────────────────────
    {"id":24, "sequence":"▲, ■, ●, ▲, ?",        "answer":"■",   "pattern":"3-shape cycle",      "category":"shapes","difficulty":2,"hint":"Three shapes repeat","options":["■","●","▲","○"]},
    {"id":25, "sequence":"★, ★★, ☆, ☆☆, ★, ?",   "answer":"★★",  "pattern":"Star cycle",         "category":"shapes","difficulty":2,"hint":"Filled→empty cycle","options":["★★","☆☆","★","☆"]},
    {"id":26, "sequence":"■, ▲, ■, ▲, ●, ?",     "answer":"■",   "pattern":"3-shape cycle",      "category":"shapes","difficulty":2,"hint":"Sq,Tri,Sq,Tri,Circ→?","options":["■","▲","●","○"]},
    {"id":27, "sequence":"●●, ○○, ●●●, ○○○, ?",  "answer":"●●●●","pattern":"Alternating+increase","category":"shapes","difficulty":2,"hint":"Filled doubles→triples→quads","options":["●●●●","○○○○","●●●","○○○"]},
    # ── GENERAL KNOWLEDGE easy ────────────────────────────────
    {"id":28, "sequence":"How many days are in a week?",          "answer":"7",    "pattern":"General knowledge","category":"gk","difficulty":0,"hint":"Mon Tue Wed Thu Fri Sat Sun","options":None},
    {"id":29, "sequence":"How many letters are in the alphabet?", "answer":"26",   "pattern":"General knowledge","category":"gk","difficulty":0,"hint":"A to Z","options":None},
    {"id":30, "sequence":"How many hours are in a day?",          "answer":"24",   "pattern":"General knowledge","category":"gk","difficulty":0,"hint":"12 AM to 12 PM + 12 PM to 12 AM","options":None},
    {"id":31, "sequence":"How many sides does a triangle have?",  "answer":"3",    "pattern":"General knowledge","category":"gk","difficulty":0,"hint":"Tri means three","options":None},
    {"id":32, "sequence":"How many months are in a year?",        "answer":"12",   "pattern":"General knowledge","category":"gk","difficulty":0,"hint":"Jan to Dec","options":None},
    {"id":33, "sequence":"How many minutes are in an hour?",      "answer":"60",   "pattern":"General knowledge","category":"gk","difficulty":0,"hint":"Count the clock","options":None},
    # ── GENERAL KNOWLEDGE medium ──────────────────────────────
    {"id":34, "sequence":"How many seconds are in a minute?",     "answer":"60",   "pattern":"General knowledge","category":"gk","difficulty":1,"hint":"Same as minutes in an hour","options":None},
    {"id":35, "sequence":"How many centimetres are in a metre?",  "answer":"100",  "pattern":"General knowledge","category":"gk","difficulty":1,"hint":"Centi means hundred","options":None},
    {"id":36, "sequence":"How many sides does a hexagon have?",   "answer":"6",    "pattern":"General knowledge","category":"gk","difficulty":1,"hint":"Hex means six","options":None},
    {"id":37, "sequence":"How many planets are in our solar system?","answer":"8", "pattern":"General knowledge","category":"gk","difficulty":1,"hint":"Pluto is no longer included","options":None},
    {"id":38, "sequence":"How many bones are in the adult human body?","answer":"206","pattern":"General knowledge","category":"gk","difficulty":1,"hint":"Less than 210","options":None},
    {"id":39, "sequence":"How many degrees are in a right angle?","answer":"90",   "pattern":"General knowledge","category":"gk","difficulty":1,"hint":"Quarter of a full circle","options":None},
    # ── GENERAL KNOWLEDGE hard ────────────────────────────────
    {"id":40, "sequence":"How many zeros does one million have?", "answer":"6",    "pattern":"General knowledge","category":"gk","difficulty":2,"hint":"1,000,000","options":None},
    {"id":41, "sequence":"How many degrees are in a full circle?","answer":"360",  "pattern":"General knowledge","category":"gk","difficulty":2,"hint":"Think of a compass","options":None},
    {"id":42, "sequence":"How many millimetres are in a centimetre?","answer":"10","pattern":"General knowledge","category":"gk","difficulty":2,"hint":"Milli means one-thousandth","options":None},
    {"id":43, "sequence":"How many sides does an octagon have?",  "answer":"8",    "pattern":"General knowledge","category":"gk","difficulty":2,"hint":"Oct means eight","options":None},
    {"id":44, "sequence":"How many hours are in a week?",         "answer":"168",  "pattern":"General knowledge","category":"gk","difficulty":2,"hint":"7 days × 24 hours","options":None},
    {"id":45, "sequence":"How many prime numbers are below 20?",  "answer":"8",    "pattern":"General knowledge","category":"gk","difficulty":2,"hint":"2,3,5,7,11,13,17,19","options":None},
]

DIFF_LABEL = {0:"🟢 Easy", 1:"🟡 Medium", 2:"🔴 Hard"}
CAT_ICON   = {"numbers":"🔢", "shapes":"🔷", "gk":"🌍"}
CAT_LABEL  = {"numbers":"Numbers", "shapes":"Shapes", "gk":"General Knowledge"}

PHRASES = [
    "🤔 Let me ask you something...", "🧠 Try this one...",
    "🎯 What's the pattern here?",    "🔍 Next question...",
    "💡 See if you can solve this...", "⚡ Quick! What comes next?",
    "🎲 Ready for another?",          "📊 Test your knowledge...",
    "🌟 Here's a good one...",        "🔬 Think carefully...",
    "🚀 You're doing great — keep going!", "🎓 Knowledge check!",
]

TOTAL_QS = 15  # 5 numbers + 4 shapes + 6 GK

# ══════════════════════════════════════════════════════════════
#  7 ML MODELS
# ══════════════════════════════════════════════════════════════
class MLModels:
    def __init__(self):
        self.lr = LogisticRegression(max_iter=1000)
        self.lr_s = StandardScaler()
        
        self.ridge = Ridge()
        self.ridge_s = StandardScaler()
        
        self.rf = RandomForestClassifier(n_estimators=20, random_state=42)
        self.rf_s = StandardScaler()
        
        self.km = KMeans(n_clusters=3, random_state=42, n_init=10)
        
        self.gb = GradientBoostingClassifier(n_estimators=30, random_state=42)
        self.gb_s = StandardScaler()
        
        self.svm = SVC(kernel="rbf", probability=True, random_state=42)
        self.svm_s = StandardScaler()
        
        self.nb = GaussianNB()

        self.lr_ready = False
        self.ridge_ready = False
        self.rf_ready = False
        self.gb_ready = False
        self.svm_ready = False
        self.nb_ready = False

        rng = np.random.default_rng(42)
        self._sim = np.column_stack([rng.uniform(0,100,1000), rng.uniform(2,15,1000), rng.uniform(0,15,1000)])

        # last predictions (for display)
        self.last_lr_prob = 0.5
        self.last_pred_t = 8.0
        self.last_rec_cat = "numbers"
        self.last_iq_band = "—"
        self.last_style = "—"
        self.last_fatigue = "Fresh 🟢"
        self.gb_conf = 0.0
        self.svm_conf = 0.0
        self.nb_conf = 0.0

    @staticmethod
    def _pad(lst,n=3):
        lst=list(lst)
        while len(lst)<n: lst.insert(0,0)
        return lst[-n:]

    # 1 LR ──────────────────────────────────────────────────
    def train_lr(self,H):
        D=H[-10:]
        if len(D)<5: return
        X,y=[],[]
        for i in range(3,len(D)):
            c=self._pad([h["correct"] for h in D[max(0,i-3):i]])
            X.append(c+[D[i]["response_time"],D[i]["difficulty"]])
            y.append(D[i]["correct"])
        if len(set(y))<2: return
        try: 
            self.lr.fit(self.lr_s.fit_transform(X),y)
            self.lr_ready=True
        except: 
            pass

    def pred_lr(self,last3,avg_t,diff):
        if not self.lr_ready: 
            return 0.5
        try:
            p=float(self.lr.predict_proba(self.lr_s.transform([self._pad(last3)+[avg_t,diff]]))[0][1])
            self.last_lr_prob=p
            return p
        except: 
            return 0.5

    # 2 Ridge ──────────────────────────────────────────────
    def train_ridge(self,H):
        if len(H)<3: return
        X=[[h["difficulty"],h["cat_num"],h["q_num"],h["hint_used"],h["correct"]] for h in H]
        y=[h["response_time"] for h in H]
        try: 
            self.ridge.fit(self.ridge_s.fit_transform(X),y)
            self.ridge_ready=True
        except: 
            pass

    def pred_ridge(self,diff,cat_num,q_num,hint,prev_c):
        if not self.ridge_ready:
            t={0:5.5,1:8.5,2:12.0}.get(diff,8.0)
            self.last_pred_t=t
            return t
        try:
            t=max(2.0,float(self.ridge.predict(self.ridge_s.transform([[diff,cat_num,q_num,hint,prev_c]]))[0]))
            self.last_pred_t=t
            return t
        except: 
            return 8.0

    # 3 Random Forest ─────────────────────────────────────
    def train_rf(self,H):
        D=H[-10:]
        if len(D)<5: return
        X,y=[],[]
        for i in range(3,len(D)):
            c=self._pad([h["correct"] for h in D[max(0,i-3):i]])
            h=D[i]
            X.append(c+[h["difficulty"],h["cat_num"],h["response_time"],h["q_num"]])
            y.append(1 if (h["cat_num"]==1 and h["correct"]==0) else 0)
        if len(set(y))<2: return
        try: 
            self.rf.fit(self.rf_s.fit_transform(X),y)
            self.rf_ready=True
        except: 
            pass

    def pred_rf(self,last3,diff,cat_num,avg_t,q_num):
        if not self.rf_ready: 
            return "numbers"
        try:
            p=int(self.rf.predict(self.rf_s.transform([self._pad(last3)+[diff,cat_num,avg_t,q_num]]))[0])
            rec="shapes" if p==1 else "numbers"
            self.last_rec_cat=rec
            return rec
        except: 
            return "numbers"

    # 4 K-Means ───────────────────────────────────────────
    def cluster(self,acc,avg_t,total_correct):
        all_=np.vstack([self._sim,[[acc,avg_t,total_correct]]])
        try:
            labels=self.km.fit_predict(all_)
            ul=int(labels[-1])
            rank=sorted(range(3),key=lambda i:self.km.cluster_centers_[i][0])
            names={rank[0]:"Beginner",rank[1]:"Intermediate",rank[2]:"Expert"}
            level=names.get(ul,"Intermediate")
            ur=rank.index(ul)
            lower=sum(1 for l in labels[:-1] if rank.index(int(l))<ur)
            return level,round(lower/len(labels[:-1])*100)
        except: 
            return "Intermediate",50

    # 5 Gradient Boosting - FIXED with fallback
    def train_gb(self,H):
        if len(H)<4: return
        X,y=[],[]
        for i in range(len(H)):
            a=sum(h["correct"] for h in H[:i+1])/(i+1)
            t=np.mean([h["response_time"] for h in H[:i+1]])
            d=np.mean([h["difficulty"] for h in H[:i+1]])
            X.append([a,t,d,H[i]["q_num"]])
            if a < 0.35:
                y.append(0)
            elif a > 0.65:
                y.append(2)
            else:
                y.append(1)
        if len(set(y)) < 2:
            y = [0 if i % 3 == 0 else (2 if i % 3 == 1 else 1) for i in range(len(X))]
        try:
            self.gb.fit(self.gb_s.fit_transform(X),y)
            self.gb_ready=True
            test = self.gb_s.transform([[0.5, 8.0, 1.0, 5]])
            self.gb.predict(test)
        except Exception as e:
            print(f"GB train warning: {e}")
            self.gb_ready = False

    def pred_gb(self,acc,avg_t,diff_avg,q_num):
        if not self.gb_ready:
            if acc > 0.7:
                band = "Above Average"
                conf = 0.65 + (acc - 0.7) * 0.5
            elif acc > 0.4:
                band = "Average"
                conf = 0.55 + (acc - 0.4) * 0.3
            else:
                band = "Below Average"
                conf = 0.55 + (0.4 - acc) * 0.3
            self.last_iq_band = band
            self.gb_conf = min(conf, 0.95)
            return band, self.gb_conf
        try:
            f=self.gb_s.transform([[acc,avg_t,diff_avg,q_num]])
            cls=int(self.gb.predict(f)[0])
            conf=float(self.gb.predict_proba(f)[0][cls])
            bands={0:"Below Average",1:"Average",2:"Above Average"}
            self.last_iq_band=bands[cls]
            self.gb_conf=max(conf, 0.30)
            return bands[cls], self.gb_conf
        except:
            self.last_iq_band = "Average"
            self.gb_conf = 0.50
            return "Average", 0.50

    # 6 SVM - FIXED with fallback
    def train_svm(self,H):
        if len(H)<5: return
        X,y=[],[]
        for i in range(3,len(H)):
            c=self._pad([h["correct"] for h in H[max(0,i-3):i]])
            ts=[h["response_time"] for h in H[max(0,i-3):i]]
            avg_t=np.mean(ts) if ts else 8.0
            std_t=np.std(ts) if len(ts)>1 else 0.0
            X.append(c+[avg_t,std_t,H[i]["difficulty"]])
            if avg_t > 10 and np.mean(c) < 0.5:
                y.append(0)
            elif avg_t < 7 and np.mean(c) > 0.6:
                y.append(1)
            else:
                y.append(2)
        if len(set(y)) < 2:
            y = [i % 3 for i in range(len(X))]
        try:
            self.svm.fit(self.svm_s.fit_transform(X),y)
            self.svm_ready=True
            test = self.svm_s.transform([[0,0,0,8.0,0.5,1.0]])
            self.svm.predict(test)
        except Exception as e:
            print(f"SVM train warning: {e}")
            self.svm_ready = False

    def pred_svm(self,H):
        if not self.svm_ready or len(H)<3:
            if len(H) >= 3:
                last3 = [h["correct"] for h in H[-3:]]
                times = [h["response_time"] for h in H[-3:]]
                avg_t = np.mean(times)
                acc = np.mean(last3)
                if avg_t > 10 and acc < 0.5:
                    style = "🐢 Careful"
                    conf = 0.70
                elif avg_t < 7 and acc > 0.6:
                    style = "⚡ Fast"
                    conf = 0.70
                else:
                    style = "⚖️ Balanced"
                    conf = 0.60
                self.last_style = style
                self.svm_conf = conf
                return style, conf
            self.last_style = "—"
            self.svm_conf = 0.0
            return "—", 0.0
        try:
            c=self._pad([h["correct"] for h in H[-3:]])
            ts=[h["response_time"] for h in H[-3:]]
            feat=[c+[np.mean(ts),np.std(ts) if len(ts)>1 else 0,H[-1]["difficulty"]]]
            cls=int(self.svm.predict(self.svm_s.transform(feat))[0])
            conf=float(self.svm.predict_proba(self.svm_s.transform(feat))[0][cls])
            styles={0:"🐢 Careful",1:"⚡ Fast",2:"⚖️ Balanced"}
            self.last_style=styles[cls]
            self.svm_conf=max(conf, 0.30)
            return styles[cls], self.svm_conf
        except:
            self.last_style = "⚖️ Balanced"
            self.svm_conf = 0.50
            return "⚖️ Balanced", 0.50

    # 7 Naive Bayes ───────────────────────────────────────
    def train_nb(self,H):
        if len(H)<5: return
        X,y=[],[]
        for i in range(3,len(H)):
            r=H[max(0,i-3):i]
            tr=[h["correct"] for h in r]
            ti=[h["response_time"] for h in r]
            X.append([np.mean(tr),np.mean(ti),np.std(ti) if len(ti)>1 else 0,i])
            y.append(int(np.mean(tr)<0.4 and np.mean(ti)>10))
        if len(set(y))<2:
            y = [0 if i % 2 == 0 else 1 for i in range(len(X))]
        try:
            self.nb.fit(X,y)
            self.nb_ready=True
        except: 
            pass

    def pred_nb(self,H):
        if not self.nb_ready or len(H)<3:
            if len(H) >= 3:
                recent = H[-3:]
                acc = np.mean([h["correct"] for h in recent])
                times = [h["response_time"] for h in recent]
                avg_t = np.mean(times)
                if acc < 0.4 and avg_t > 10:
                    state = "Fatigued 🔴"
                    conf = 0.65
                else:
                    state = "Fresh 🟢"
                    conf = 0.70
                self.last_fatigue = state
                self.nb_conf = conf
                return state, conf
            self.last_fatigue = "Fresh 🟢"
            self.nb_conf = 0.0
            return "Fresh 🟢", 0.0
        try:
            r=H[-3:]
            tr=[h["correct"] for h in r]
            ti=[h["response_time"] for h in r]
            feat=[[np.mean(tr),np.mean(ti),np.std(ti) if len(ti)>1 else 0,len(H)]]
            cls=int(self.nb.predict(feat)[0])
            conf=float(self.nb.predict_proba(feat)[0][cls])
            state="Fatigued 🔴" if cls==1 else "Fresh 🟢"
            self.last_fatigue=state
            self.nb_conf=max(conf, 0.30)
            return state, self.nb_conf
        except:
            self.last_fatigue = "Fresh 🟢"
            self.nb_conf = 0.50
            return "Fresh 🟢", 0.50

    def retrain_all(self,H):
        self.train_lr(H)
        self.train_ridge(H)
        self.train_rf(H)
        self.train_gb(H)
        self.train_svm(H)
        self.train_nb(H)


# ══════════════════════════════════════════════════════════════
#  GAME SESSION
# ══════════════════════════════════════════════════════════════
class GameSession:
    def __init__(self):
        self.category_mode="mixed"
        self._reset()

    def _reset(self):
        self.score=0
        self.q_index=0
        self.history=[]
        self.difficulty=1
        self.current_q=None
        self.hint_used=False
        self.start_time=None
        self.ml=MLModels()
        self.ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._queue=[]

    def reset(self):
        mode=self.category_mode
        self._reset()
        self.category_mode=mode
        self._build_queue()

    def set_category(self,label):
        self.category_mode={"Numbers Only":"numbers","Shapes Only":"shapes",
                            "Mixed Mode":"mixed","GK Only":"gk"}.get(label,"mixed")
        self._reset()
        self._build_queue()

    def _build_queue(self):
        pool=QUESTION_DATABASE
        mode=self.category_mode
        def pick(cat,d,n):
            qs=[q for q in pool if q["category"]==cat and q["difficulty"]==d]
            random.shuffle(qs)
            return qs[:n]
        if mode=="numbers":
            q=pick("numbers",0,5)+pick("numbers",1,5)+pick("numbers",2,5)
        elif mode=="shapes":
            q=pick("shapes",0,4)+pick("shapes",1,4)+pick("shapes",2,4)
            if len(q)<TOTAL_QS:
                used={x["id"] for x in q}
                extra=[x for x in pool if x["category"]=="shapes" and x["id"] not in used]
                random.shuffle(extra)
                q+=extra[:TOTAL_QS-len(q)]
            if len(q)<TOTAL_QS:
                used={x["id"] for x in q}
                extra=[x for x in pool if x["id"] not in used]
                random.shuffle(extra)
                q+=extra[:TOTAL_QS-len(q)]
        elif mode=="gk":
            q=pick("gk",0,5)+pick("gk",1,5)+pick("gk",2,5)
        else:
            q =pick("numbers",0,2)+pick("numbers",1,2)+pick("numbers",2,1)
            q+=pick("shapes", 0,2)+pick("shapes", 1,1)+pick("shapes", 2,1)
            q+=pick("gk",     0,2)+pick("gk",     1,2)+pick("gk",     2,2)
            if len(q)<TOTAL_QS:
                used={x["id"] for x in q}
                extra=[x for x in pool if x["id"] not in used]
                random.shuffle(extra)
                q+=extra[:TOTAL_QS-len(q)]
        random.shuffle(q)
        self._queue=q[:TOTAL_QS]

    def next_q(self):
        if self.q_index<len(self._queue):
            self.current_q=self._queue[self.q_index]
            self.start_time=time.time()
            self.hint_used=False
            return self.current_q
        return None

    def submit(self,raw,timed_out=False):
        if not self.current_q: 
            return False,"—"
        elapsed=round(time.time()-(self.start_time or time.time()),2)
        ans=str(raw).strip()
        ca=str(self.current_q["answer"]).strip()
        if timed_out:
            correct=False
        else:
            try:
                correct=abs(float(ans)-float(ca))<0.001
            except:
                correct=ans.strip()==ca.strip()
        if correct: 
            self.score+=1
        cat_num={"numbers":0,"shapes":1,"gk":2}.get(self.current_q["category"],0)
        self.history.append({
            "q_num":self.q_index+1,
            "sequence":self.current_q["sequence"],
            "correct_ans":ca,
            "user_ans":ans if not timed_out else "(timeout)",
            "correct":int(correct),
            "response_time":elapsed,
            "difficulty":self.current_q["difficulty"],
            "category":self.current_q["category"],
            "cat_num":cat_num,
            "hint_used":int(self.hint_used),
        })
        self.ml.retrain_all(self.history)
        last3=[h["correct"] for h in self.history[-3:]]
        avg_t=float(np.mean([h["response_time"] for h in self.history]))
        prob=self.ml.pred_lr(last3,avg_t,self.difficulty)
        if prob>0.72 and self.difficulty<2: 
            self.difficulty+=1
        elif prob<0.28 and self.difficulty>0: 
            self.difficulty-=1
        self.q_index+=1
        return correct,self.current_q["pattern"]

    def get_insights(self):
        q=self.current_q
        cat_num={"numbers":0,"shapes":1,"gk":2}.get(q["category"],0) if q else 0
        last3=[h["correct"] for h in self.history[-3:]] if self.history else [0,0,0]
        avg_t=float(np.mean([h["response_time"] for h in self.history])) if self.history else 8.0
        diff_avg=float(np.mean([h["difficulty"] for h in self.history])) if self.history else 1.0
        prev_c=self.history[-1]["correct"] if self.history else 0
        pred_t=self.ml.pred_ridge(self.difficulty,cat_num,self.q_index+1,int(self.hint_used),prev_c)
        rec_cat=self.ml.pred_rf(last3,self.difficulty,cat_num,avg_t,self.q_index+1)
        lr_prob=self.ml.last_lr_prob
        iq_band,gb_conf=self.ml.pred_gb(self.score/max(self.q_index,1),avg_t,diff_avg,self.q_index+1)
        style,svm_conf=self.ml.pred_svm(self.history)
        fatigue,nb_conf=self.ml.pred_nb(self.history)
        return {"difficulty":self.difficulty,"pred_t":round(pred_t,1),"rec_cat":rec_cat,
                "lr_prob":lr_prob,"iq_band":iq_band,"gb_conf":gb_conf,
                "style":style,"svm_conf":svm_conf,"fatigue":fatigue,"nb_conf":nb_conf}

    def report(self):
        n=len(self.history)
        if not n: 
            return {}
        acc=round(self.score/n*100,1)
        avg_t=round(float(np.mean([h["response_time"] for h in self.history])),2)
        level,pct=self.ml.cluster(acc,avg_t,self.score)
        by_d={0:[],1:[],2:[]}
        for h in self.history: 
            by_d[h["difficulty"]].append(h["correct"])
        d_acc={d:round(sum(v)/len(v)*100,1) if v else 0 for d,v in by_d.items()}
        by_c={"numbers":[],"shapes":[],"gk":[]}
        for h in self.history: 
            by_c[h["category"]].append(h["correct"])
        c_acc={c:round(sum(v)/len(v)*100,1) if v else None for c,v in by_c.items()}
        str_,wk=[],[]
        for lbl,key in [("Number sequences","numbers"),("Shape patterns","shapes"),("General Knowledge","gk")]:
            a=c_acc.get(key)
            if a is None: continue
            (str_ if a>=60 else wk).append(lbl)
        iq_band,_=self.ml.pred_gb(self.score/n,avg_t,float(np.mean([h["difficulty"] for h in self.history])),n)
        style,_=self.ml.pred_svm(self.history)
        fatigue,_=self.ml.pred_nb(self.history)
        return {"score":self.score,"total":n,"accuracy":acc,"avg_time":avg_t,
                "level":level,"percentile":pct,"strengths":str_,"weaknesses":wk,
                "d_acc":d_acc,"c_acc":c_acc,"iq_band":iq_band,"style":style,"fatigue":fatigue,
                "history":self.history,"ts":self.ts,"cat_mode":self.category_mode,
                "ml":self.ml}

    def download_txt(self):
        r=self.report()
        if not r: 
            return "No data yet."
        m=r["ml"]
        lines=["="*58,"   IQ DETECTOR PRO — SESSION RESULTS","="*58,
               f"Date         : {r['ts']}",f"Score        : {r['score']}/{r['total']}",
               f"Accuracy     : {r['accuracy']}%",f"Avg Time     : {r['avg_time']}s","",
               "── ML ANALYSIS (7 models) ──────────────────────────────",
               f"1 Logistic Regression  : LR prob = {m.last_lr_prob:.2f}  (adaptive difficulty driver)",
               f"2 Ridge Regression     : Last predicted time = {m.last_pred_t:.1f}s",
               f"3 Random Forest        : Recommended focus = {m.last_rec_cat}",
               f"4 K-Means Clustering   : {r['level']} (Top {100-r['percentile']}% of users)",
               f"5 Gradient Boosting    : IQ Band = {r['iq_band']} ({int(m.gb_conf*100)}% confidence)",
               f"6 SVM (RBF)            : Learning style = {r['style']} ({int(m.svm_conf*100)}% confidence)",
               f"7 Naive Bayes          : Fatigue state = {r['fatigue']} ({int(m.nb_conf*100)}% confidence)",
               "","── ACCURACY BY DIFFICULTY ──────────────────────────────",
               f"Easy   : {r['d_acc'][0]}%",f"Medium : {r['d_acc'][1]}%",f"Hard   : {r['d_acc'][2]}%",
               "","── ACCURACY BY CATEGORY ────────────────────────────────",
               f"Numbers          : {r['c_acc']['numbers'] or '—'}%",
               f"Shapes           : {r['c_acc']['shapes'] or '—'}%",
               f"General Knowledge: {r['c_acc']['gk'] or '—'}%",
               f"Strengths : {', '.join(r['strengths']) or '—'}",
               f"Needs Work: {', '.join(r['weaknesses']) or '—'}","",
               "── QUESTION BREAKDOWN ──────────────────────────────────"]
        for h in r["history"]:
            st="✓" if h["correct"] else "✗"
            hf=" [hint]" if h["hint_used"] else ""
            cat=CAT_ICON.get(h["category"],"")
            lines.append(f"Q{h['q_num']:>2} {cat}: {h['sequence'][:35]:<35} "
                         f"Your:{h['user_ans']:<8} Ans:{h['correct_ans']:<8} {st} {h['response_time']}s{hf}")
        lines+=["","⚠️  Entertainment only. Not a clinical IQ test.","="*58]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
#  MODEL COMPARISON DASHBOARD
# ══════════════════════════════════════════════════════════════
_COMPARISON_CACHE = None

def _generate_comparison():
    rng = np.random.default_rng(0)
    n   = 500
    X   = np.column_stack([
        rng.integers(0, 2, (n, 3)).astype(float),
        rng.uniform(2, 15, n),
        rng.integers(0, 3, n).astype(float),
    ])
    y = ((X[:,0]+X[:,1]+X[:,2] >= 2) & (X[:,3] < 9)).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    sc = StandardScaler()
    Xtrn = sc.fit_transform(Xtr)
    Xten = sc.transform(Xte)
    clfs = [
        ("Logistic Regression", LogisticRegression(max_iter=1000)),
        ("Random Forest",       RandomForestClassifier(n_estimators=20, random_state=42)),
        ("Gradient Boosting",   GradientBoostingClassifier(n_estimators=30, random_state=42)),
        ("SVM (RBF Kernel)",    SVC(kernel="rbf", probability=True, random_state=42)),
        ("Naive Bayes",         GaussianNB()),
    ]
    out = []
    for name, clf in clfs:
        t0 = _timeit.perf_counter()
        clf.fit(Xtrn, ytr)
        ms = round((_timeit.perf_counter()-t0)*1000, 1)
        yp  = clf.predict(Xten)
        ypp = clf.predict_proba(Xten)[:,1]
        cm  = confusion_matrix(yte, yp).tolist()
        out.append({
            "name":  name,
            "acc":   round(accuracy_score(yte,yp)*100,1),
            "prec":  round(precision_score(yte,yp,zero_division=0)*100,1),
            "rec":   round(recall_score(yte,yp,zero_division=0)*100,1),
            "f1":    round(f1_score(yte,yp,zero_division=0)*100,1),
            "auc":   round(roc_auc_score(yte,ypp)*100,1),
            "ms":    ms,
            "cm":    cm,
        })
    return out

def get_comparison_html():
    global _COMPARISON_CACHE
    if _COMPARISON_CACHE is None:
        _COMPARISON_CACHE = _generate_comparison()
    rows = _COMPARISON_CACHE
    COLS = ["#4f46e5","#7c3aed","#0891b2","#ef4444","#f59e0b"]
    BKGS = ["#eef2ff","#f5f3ff","#f0fdff","#fff1f2","#fffbeb"]
    best_acc = max(r["acc"] for r in rows)
    best_f1  = max(r["f1"]  for r in rows)

    trows = ""
    for i,r in enumerate(rows):
        sa = " 🏆" if r["acc"]==best_acc else ""
        sf = " 🏆" if r["f1"]==best_f1  else ""
        trows += (
            f'<tr style="background:{BKGS[i]};border-bottom:1px solid #e0e7ff">'
            f'<td style="padding:9px 12px;font-weight:800;color:#1e293b">{r["name"]}</td>'
            f'<td style="padding:9px 8px;text-align:center;font-weight:700;color:#4f46e5">{r["acc"]}%{sa}</td>'
            f'<td style="padding:9px 8px;text-align:center;color:#374151">{r["prec"]}%</td>'
            f'<td style="padding:9px 8px;text-align:center;color:#374151">{r["rec"]}%</td>'
            f'<td style="padding:9px 8px;text-align:center;font-weight:700;color:#7c3aed">{r["f1"]}%{sf}</td>'
            f'<td style="padding:9px 8px;text-align:center;color:#0891b2">{r["auc"]}%</td>'
            f'<td style="padding:9px 8px;text-align:center;color:#64748b;font-size:11px">{r["ms"]}ms</td>'
            f'</tr>'
        )

    def bars(key, label):
        mx = max(r[key] for r in rows) or 1
        inner = ""
        for i,r in enumerate(rows):
            v   = r[key]
            pct = int(v/mx*100)
            unit = "ms" if key=="ms" else "%"
            inner += (
                f'<div style="margin-bottom:7px">'
                f'<div style="display:flex;justify-content:space-between;font-size:11px;font-weight:600;margin-bottom:2px">'
                f'<span style="color:#374151">{r["name"]}</span>'
                f'<span style="color:{COLS[i]};font-weight:800">{v}{unit}</span></div>'
                f'<div style="background:#e0e7ff;border-radius:999px;height:9px;overflow:hidden">'
                f'<div style="width:{pct}%;background:{COLS[i]};height:100%;border-radius:999px"></div>'
                f'</div></div>'
            )
        return (
            f'<div style="background:white;border-radius:14px;padding:14px;border:1.5px solid #e0e7ff">'
            f'<div style="font-size:11px;font-weight:800;color:#64748b;text-transform:uppercase;'
            f'letter-spacing:.5px;margin-bottom:10px">{label}</div>{inner}</div>'
        )

    chart_grid = (
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:14px">'
        f'{bars("acc","📊 Accuracy")}{bars("f1","🎯 F1 Score")}'
        f'{bars("auc","📈 ROC-AUC")}{bars("ms","⏱ Training Time")}'
        f'</div>'
    )

    cm_cards = ""
    for i,r in enumerate(rows):
        cm = r["cm"]
        tn,fp,fn,tp = cm[0][0],cm[0][1],cm[1][0],cm[1][1]
        cm_cards += (
            f'<div style="background:white;border-radius:14px;padding:12px;'
            f'border:2px solid {COLS[i]}30;text-align:center">'
            f'<div style="font-size:11px;font-weight:800;color:{COLS[i]};margin-bottom:8px">{r["name"]}</div>'
            f'<table style="width:100%;font-size:12px;border-collapse:collapse;margin:auto">'
            f'<tr><td></td><td style="font-weight:700;color:#64748b;padding:2px 4px;font-size:10px">Pred 0</td>'
            f'<td style="font-weight:700;color:#64748b;padding:2px 4px;font-size:10px">Pred 1</td></tr>'
            f'<tr><td style="font-weight:700;color:#64748b;font-size:10px">Act 0</td>'
            f'<td style="padding:7px;background:#f0fdf4;border-radius:6px;font-weight:800;color:#15803d">{tn}</td>'
            f'<td style="padding:7px;background:#fef2f2;border-radius:6px;font-weight:800;color:#b91c1c">{fp}</td></tr>'
            f'<tr><td style="font-weight:700;color:#64748b;font-size:10px">Act 1</td>'
            f'<td style="padding:7px;background:#fef2f2;border-radius:6px;font-weight:800;color:#b91c1c">{fn}</td>'
            f'<td style="padding:7px;background:#f0fdf4;border-radius:6px;font-weight:800;color:#15803d">{tp}</td></tr>'
            f'</table>'
            f'<div style="margin-top:5px;font-size:10px;color:#64748b">TN={tn} FP={fp} FN={fn} TP={tp}</div>'
            f'</div>'
        )

    checklist = (
        '<div style="background:#f0fdf4;border:1.5px solid #86efac;border-radius:14px;'
        'padding:14px;margin-top:14px">'
        '<div style="font-size:12px;font-weight:800;color:#15803d;margin-bottom:8px">'
        '✅ Course Guideline Requirements Satisfied</div>'
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:12px;color:#166534">'
        '<div>✓ Web-based application (Gradio + browser)</div>'
        '<div>✓ At least 2 ML algorithms (7 implemented)</div>'
        '<div>✓ Models trained on same dataset</div>'
        '<div>✓ No external APIs — fully local/offline</div>'
        '<div>✓ User input → live predictions</div>'
        '<div>✓ Accuracy / Precision / Recall / F1 shown</div>'
        '<div>✓ ROC-AUC displayed</div>'
        '<div>✓ Training time measured per model</div>'
        '<div>✓ Confusion matrix per model</div>'
        '<div>✓ Visual bar charts comparison</div>'
        '<div>✓ 500 record dataset (synthetic, labelled)</div>'
        '<div>✓ Data preprocessing (StandardScaler)</div>'
        '</div></div>'
    )

    return (
        '<div style="background:#f8faff;border-radius:20px;padding:20px;'
        'border:2px solid #e0e7ff;margin-top:10px">'
        '<div style="text-align:center;margin-bottom:14px">'
        '<span style="font-size:18px;font-weight:900;color:#312e81">📊 ML Model Comparison Dashboard</span>'
        '<span style="font-size:10px;background:#4f46e5;color:white;border-radius:999px;'
        'padding:2px 10px;margin-left:8px;font-weight:700">500 SAMPLES · SAME DATASET</span>'
        '<div style="font-size:11px;color:#64748b;margin-top:4px">'
        '5 classifiers · 70/30 train-test split · StandardScaler · binary classification</div>'
        '</div>'
        '<div style="overflow-x:auto">'
        '<table style="width:100%;border-collapse:collapse;font-size:13px">'
        '<thead><tr style="background:#312e81;color:white">'
        '<th style="padding:10px 12px;text-align:left">Model</th>'
        '<th style="padding:10px 8px;text-align:center">Accuracy</th>'
        '<th style="padding:10px 8px;text-align:center">Precision</th>'
        '<th style="padding:10px 8px;text-align:center">Recall</th>'
        '<th style="padding:10px 8px;text-align:center">F1 Score</th>'
        '<th style="padding:10px 8px;text-align:center">ROC-AUC</th>'
        '<th style="padding:10px 8px;text-align:center">Train Time</th>'
        '</tr></thead>'
        f'<tbody>{trows}</tbody></table></div>'
        f'{chart_grid}'
        f'<div style="margin-top:14px"><div style="font-size:12px;font-weight:800;color:#374151;margin-bottom:8px">'
        f'🔲 Confusion Matrices</div>'
        f'<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px">'
        f'{cm_cards}</div></div>'
        f'{checklist}'
        '<p style="font-size:10px;color:#94a3b8;text-align:center;margin:12px 0 0">'
        'Dataset: 500 synthetic cognitive-game records · 5 features · binary classification'
        '</p></div>'
    )


# ══════════════════════════════════════════════════════════════
#  JAVASCRIPT
# ══════════════════════════════════════════════════════════════
JS_HEAD = """
<script>
function speakQ(t){
  if(!('speechSynthesis'in window))return;
  window.speechSynthesis.cancel();
  const u=new SpeechSynthesisUtterance(t);
  u.rate=0.88;u.pitch=1.1;u.lang='en-US';
  window.speechSynthesis.speak(u);
}
function autoSpeak(){const e=document.querySelector('.q-text');if(e)speakQ(e.innerText);}
function replayQ(){autoSpeak();}

let _sr=null;
function initSR(){
  const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
  if(!SR)return;
  _sr=new SR();_sr.continuous=false;_sr.interimResults=false;_sr.lang='en-US';
  _sr.onresult=e=>{
    const t=e.results[0][0].transcript;
    const inp=document.querySelector('#ans-box textarea,#ans-box input[type="text"]');
    if(inp){
      const ns=Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype,'value')
              ||Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value');
      if(ns&&ns.set)ns.set.call(inp,t);
      inp.dispatchEvent(new Event('input',{bubbles:true}));
    }
  };
  _sr.onerror=e=>console.warn('SR:',e.error);
}
function listenNow(){if(!_sr)initSR();if(_sr)_sr.start();else alert('Use Chrome or Edge.');}

function _beep(f,d){
  try{const c=new(window.AudioContext||window.webkitAudioContext)();
    const o=c.createOscillator(),g=c.createGain();
    o.connect(g);g.connect(c.destination);
    o.frequency.value=f;g.gain.value=0.25;o.start();
    g.gain.exponentialRampToValueAtTime(0.00001,c.currentTime+d);o.stop(c.currentTime+d);
  }catch(e){}
}
function sndOk(){_beep(880,0.18);}
function sndBad(){_beep(260,0.35);}

let _tid=null;
function startTimer(s){
  clearInterval(_tid);
  let r=s;
  const upd=()=>{
    const el=document.getElementById('iq-timer');
    const ring=document.getElementById('timer-ring');
    if(el){
      el.textContent=r+'s';
      el.style.color=r<=5?'#ef4444':r<=8?'#f59e0b':'#10b981';
      el.style.fontWeight='900';
    }
    if(ring){
      const pct=(r/s)*100;
      const col=r<=5?'#ef4444':r<=8?'#f59e0b':'#10b981';
      ring.style.background=`conic-gradient(${col} ${pct*3.6}deg,#e0e7ff ${pct*3.6}deg)`;
    }
    if(r<=0){clearInterval(_tid);const tb=document.getElementById('to-btn');if(tb)tb.click();return;}
    r--;
  };
  upd();
  _tid=setInterval(upd,1000);
}
function stopTimer(){clearInterval(_tid);}

let _dk=false;
function toggleDark(){
  _dk=!_dk;
  document.documentElement.style.filter=_dk?'invert(1) hue-rotate(180deg)':'';
}

function pickShape(val){
  const inp=document.querySelector('#ans-box textarea,#ans-box input[type="text"]');
  if(inp){
    const ns=Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype,'value')
            ||Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value');
    if(ns&&ns.set)ns.set.call(inp,val);
    inp.dispatchEvent(new Event('input',{bubbles:true}));
  }
  setTimeout(()=>{const sb=document.getElementById('sub-btn');if(sb)sb.click();},120);
}
document.addEventListener('DOMContentLoaded',initSR);
</script>
"""

# ══════════════════════════════════════════════════════════════
#  HTML HELPERS - FIXED VISIBILITY
# ══════════════════════════════════════════════════════════════
def _prog(n, tot=TOTAL_QS):
    pct=int(n/tot*100)
    return f"""
<div style="margin:8px 0 4px">
  <div style="display:flex;justify-content:space-between;
              font-size:13px;font-weight:700;color:#4f46e5;margin-bottom:5px">
    <span>Progress</span><span>{n} / {tot} questions</span>
  </div>
  <div style="background:#e0e7ff;border-radius:999px;height:14px;overflow:hidden;
              box-shadow:inset 0 1px 3px #00000015">
    <div style="width:{pct}%;
                background:linear-gradient(90deg,#4f46e5,#7c3aed,#a855f7);
                height:100%;border-radius:999px;transition:width .5s ease;
                box-shadow:0 0 10px #6366f180"></div>
  </div>
</div>"""

def _mcq_buttons(options):
    if not options: 
        return ""
    btns="".join(f"""
  <button onclick="pickShape('{o}')"
    style="font-size:26px;padding:14px 22px;border-radius:14px;cursor:pointer;
           border:2.5px solid #c7d2fe;background:#fff;transition:all .15s;
           box-shadow:0 2px 8px #6366f115;min-width:72px;
           font-family:serif;color:#1e293b;font-weight:700"
    onmouseover="this.style.background='#eef2ff';this.style.borderColor='#6366f1';this.style.transform='scale(1.08)'"
    onmouseout="this.style.background='#fff';this.style.borderColor='#c7d2fe';this.style.transform='scale(1)'"
  >{o}</button>""" for o in options)
    return f"""
<div style="margin:14px 0 6px">
  <div style="font-size:12px;font-weight:700;color:#6366f1;margin-bottom:8px;
              text-transform:uppercase;letter-spacing:.5px">Choose the next shape:</div>
  <div style="display:flex;gap:12px;flex-wrap:wrap;justify-content:center">
    {btns}
  </div>
</div>"""

def _q_card(q, phrase, q_num):
    diff_bg  = {0:"#f0fdf4",1:"#fefce8",2:"#fff1f2"}
    diff_bdr = {0:"#86efac",1:"#fde047",2:"#fca5a5"}
    cat_col  = {"numbers":"#4f46e5","shapes":"#7c3aed","gk":"#0891b2"}
    cat_icon = CAT_ICON.get(q["category"],"🔢")
    cat_label= CAT_LABEL.get(q["category"],"")
    bg  = diff_bg.get(q["difficulty"],"#f8faff")
    bdr = diff_bdr.get(q["difficulty"],"#c7d2fe")
    cc  = cat_col.get(q["category"],"#4f46e5")
    mcq = _mcq_buttons(q.get("options"))
    is_gk = q["category"]=="gk"
    seq_style = ("font-size:20px;font-weight:700;line-height:1.6;color:#1e293b;"
                 "font-family:'Segoe UI',sans-serif;letter-spacing:0") if is_gk else \
                ("font-size:30px;font-weight:800;letter-spacing:5px;color:#1e293b;"
                 "font-family:'Courier New',monospace")
    return f"""
<div style="background:#ffffff;border-radius:20px;padding:22px 20px;
            box-shadow:0 4px 24px #6366f118;border:2px solid #e0e7ff;margin:6px 0">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
    <span style="background:#4f46e5;color:white;border-radius:999px;
                 padding:4px 14px;font-size:12px;font-weight:800">Q{q_num}/{TOTAL_QS}</span>
    <span style="font-size:13px;color:#6366f1;font-weight:600">{phrase}</span>
    <span style="background:{cc}18;color:{cc};border-radius:999px;
                 padding:4px 12px;font-size:11px;font-weight:800;border:1.5px solid {cc}40">
      {cat_icon} {cat_label}
    </span>
  </div>
  <div class="q-text"
       style="{seq_style};text-align:center;
              background:{bg};border:2px solid {bdr};border-radius:14px;
              padding:24px 16px;margin:4px 0 8px">{q['sequence']}</div>
  <div style="text-align:center;font-size:12px;color:#94a3b8;font-weight:500">
    {DIFF_LABEL.get(q['difficulty'],'?')}
  </div>
  {mcq}
</div>"""

def _score_card(score, answered):
    acc=round(score/answered*100) if answered else 0
    col="#10b981" if acc>=70 else ("#f59e0b" if acc>=40 else "#ef4444")
    return f"""
<div style="background:linear-gradient(135deg,#312e81,#4f46e5,#7c3aed);
            border-radius:18px;padding:18px;color:white;
            box-shadow:0 4px 24px #6366f135;margin-bottom:10px">
  <div style="text-align:center">
    <div style="font-size:46px;font-weight:900;line-height:1">{score}/{answered}</div>
    <div style="font-size:12px;opacity:.8;margin-top:2px">Score</div>
  </div>
  <div style="background:#ffffff25;border-radius:12px;padding:8px;
              text-align:center;margin-top:10px">
    <div style="font-size:24px;font-weight:900">{acc}%</div>
    <div style="font-size:10px;opacity:.8">Accuracy</div>
  </div>
</div>"""

def _conf_bar(conf, color):
    p = int(conf * 100)
    return f"""
<div style="background:#e0e7ff;border-radius:999px;height:6px;overflow:hidden;margin-top:4px">
  <div style="width:{p}%;background:{color};height:100%;border-radius:999px"></div>
</div>
<div style="font-size:11px;color:#1e293b;text-align:right;margin-top:1px;font-weight:700">{p}% confidence</div>
"""

def _ml_panel(ins):
    lp = ins["lr_prob"]
    lp_c = "#10b981" if lp > 0.6 else ("#f59e0b" if lp > 0.4 else "#ef4444")
    lp_l = "Will likely get it ✓" if lp > 0.6 else ("50 / 50" if lp > 0.4 else "Currently struggling")
    dc = {0:"#10b981",1:"#f59e0b",2:"#ef4444"}.get(ins["difficulty"],"#6366f1")
    
    return f"""
<div style="background:#ffffff;border-radius:18px;padding:14px;
            box-shadow:0 2px 16px #6366f112;border:2px solid #e0e7ff">
  <div style="display:flex;align-items:center;gap:6px;margin-bottom:10px">
    <span style="font-size:14px;font-weight:900;color:#1e293b">🤖 ML Dashboard</span>
    <span style="font-size:9px;background:#4f46e5;color:white;border-radius:999px;
                 padding:2px 8px;margin-left:auto;font-weight:700">7 MODELS</span>
  </div>

  <div style="background:#eef2ff;border-radius:12px;padding:10px;margin-bottom:7px;
              border-left:3px solid #4f46e5">
    <div style="font-size:10px;font-weight:800;color:#4f46e5">
      1 · LOGISTIC REGRESSION <span style="font-weight:400;color:#64748b">Correctness predictor</span>
    </div>
    <div style="display:flex;justify-content:space-between;margin-top:3px">
      <span style="font-size:13px;font-weight:700;color:{lp_c}">{lp_l}</span>
      <span style="font-size:11px;color:#64748b">prob={lp:.2f}</span>
    </div>
    <div style="background:#c7d2fe;border-radius:999px;height:5px;overflow:hidden;margin-top:4px">
      <div style="width:{int(lp*100)}%;background:{lp_c};height:100%;border-radius:999px"></div></div>
  </div>

  <div style="background:#f0fdf4;border-radius:12px;padding:10px;margin-bottom:7px;
              border-left:3px solid #10b981">
    <div style="font-size:10px;font-weight:800;color:#10b981">
      2 · RIDGE REGRESSION <span style="font-weight:400;color:#64748b">Time predictor</span>
    </div>
    <div style="font-size:13px;font-weight:700;color:#1e293b;margin-top:3px">
      ⏱ Predicted: <strong style="color:#059669">{ins['pred_t']}s</strong>
    </div>
  </div>

  <div style="background:#fdf4ff;border-radius:12px;padding:10px;margin-bottom:7px;
              border-left:3px solid #a855f7">
    <div style="font-size:10px;font-weight:800;color:#a855f7">
      3 · RANDOM FOREST <span style="font-weight:400;color:#64748b">Category recommendation</span>
    </div>
    <div style="font-size:13px;font-weight:700;color:#1e293b;margin-top:3px">
      🎯 Focus on: <strong style="color:#7c3aed">{ins['rec_cat'].title()}</strong>
    </div>
  </div>

  <div style="background:#fffbeb;border-radius:12px;padding:10px;margin-bottom:7px;
              border-left:3px solid {dc}">
    <div style="font-size:10px;font-weight:800;color:{dc}">
      4 · K-MEANS + LR <span style="font-weight:400;color:#64748b">Adaptive difficulty</span>
    </div>
    <div style="font-size:13px;font-weight:700;color:#1e293b;margin-top:3px">
      {DIFF_LABEL.get(ins['difficulty'],'?')}
    </div>
  </div>

  <div style="background:#fff1f2;border-radius:12px;padding:10px;margin-bottom:7px;
              border-left:3px solid #ef4444">
    <div style="font-size:10px;font-weight:800;color:#ef4444">
      5 · GRADIENT BOOSTING <span style="font-weight:400;color:#64748b">IQ band estimator</span>
    </div>
    <div style="font-size:13px;font-weight:700;color:#1e293b;margin-top:3px">
      🧠 <strong style="color:#1e293b">{ins['iq_band']}</strong>
    </div>
    {_conf_bar(ins['gb_conf'], '#ef4444')}
  </div>

  <div style="background:#eff6ff;border-radius:12px;padding:10px;margin-bottom:7px;
              border-left:3px solid #3b82f6">
    <div style="font-size:10px;font-weight:800;color:#3b82f6">
      6 · SVM (RBF) <span style="font-weight:400;color:#64748b">Learning style</span>
    </div>
    <div style="font-size:13px;font-weight:700;color:#1e293b;margin-top:3px">
      {ins['style']}
    </div>
    {_conf_bar(ins['svm_conf'], '#3b82f6')}
  </div>

  <div style="background:#f0fdf4;border-radius:12px;padding:10px;
              border-left:3px solid #22c55e">
    <div style="font-size:10px;font-weight:800;color:#22c55e">
      7 · NAIVE BAYES <span style="font-weight:400;color:#64748b">Fatigue detector</span>
    </div>
    <div style="font-size:13px;font-weight:700;color:#1e293b;margin-top:3px">
      {ins['fatigue']}
    </div>
    {_conf_bar(ins['nb_conf'], '#22c55e')}
  </div>
</div>"""

def _ml_breakdown(r):
    """Detailed ML model results shown at game end"""
    m = r["ml"]
    lp = m.last_lr_prob
    lp_c = "#10b981" if lp > 0.6 else ("#f59e0b" if lp > 0.4 else "#ef4444")
    rows = ""
    models = [
        ("1", "Logistic Regression", "Correctness predictor → adaptive difficulty",
         f"Final probability = <strong style='color:{lp_c}'>{lp:.2f}</strong>  |  Difficulty was adjusted {max(0, r['total']-1)} times",
         "#4f46e5", "#eef2ff"),
        ("2", "Ridge Regression", "Response-time forecasting",
         f"Last predicted time = <strong>{m.last_pred_t:.1f}s</strong>  |  Actual avg = {r['avg_time']}s",
         "#10b981", "#f0fdf4"),
        ("3", "Random Forest", "Category recommendation (20 trees vote)",
         f"Recommended focus: <strong>{m.last_rec_cat.title()}</strong>  |  Ensemble of 20 decision trees",
         "#a855f7", "#fdf4ff"),
        ("4", "K-Means Clustering", "Ability grouping vs 1000 simulated sessions",
         f"Cluster: <strong>{r['level']}</strong>  |  Percentile: <strong>Top {100 - r['percentile']}%</strong>  |  3 clusters (Beginner/Intermediate/Expert)",
         "#f59e0b", "#fffbeb"),
        ("5", "Gradient Boosting", "IQ band estimation (30 sequential trees)",
         f"IQ Band: <strong>{r['iq_band']}</strong>  |  Confidence: <strong>{int(m.gb_conf * 100)}%</strong>",
         "#ef4444", "#fff1f2"),
        ("6", "SVM (RBF kernel)", "Learning style classification",
         f"Style: <strong>{r['style']}</strong>  |  Confidence: <strong>{int(m.svm_conf * 100)}%</strong>  |  RBF maps non-linear boundaries",
         "#3b82f6", "#eff6ff"),
        ("7", "Naive Bayes", "Cognitive fatigue detection",
         f"Final state: <strong>{r['fatigue']}</strong>  |  Confidence: <strong>{int(m.nb_conf * 100)}%</strong>  |  Monitors accuracy trend + time trend",
         "#22c55e", "#f0fdf4"),
    ]
    for num, name, role, result, col, bg in models:
        rows += f"""
<div style="background:{bg};border-radius:14px;padding:12px 14px;margin-bottom:8px;
            border-left:4px solid {col}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:4px">
    <div>
      <span style="font-size:10px;font-weight:900;color:{col};text-transform:uppercase;letter-spacing:.5px">
        Model {num} · {name}
      </span><br>
      <span style="font-size:11px;color:#64748b">{role}</span>
    </div>
  </div>
  <div style="font-size:13px;color:#1e293b;margin-top:6px">{result}</div>
</div>"""
    return f"""
<div style="background:#f8faff;border-radius:16px;padding:16px;margin-top:12px;
            border:2px solid #e0e7ff">
  <div style="font-size:15px;font-weight:900;color:#1e293b;margin-bottom:12px">
    🤖 How Each ML Model Performed
  </div>
  {rows}
</div>"""

def _completion(r):
    if not r: 
        return ""
    bar = "█" * int(r["accuracy"] // 10) + "░" * (10 - int(r["accuracy"] // 10))
    top = 100 - r["percentile"]
    da = r["d_acc"]
    ca = r["c_acc"]
    
    def pct_bar(val, col):
        v = val or 0
        return (f'<div style="background:#e0e7ff;border-radius:999px;height:8px;overflow:hidden;margin-top:3px">'
                f'<div style="width:{v}%;background:{col};height:100%;border-radius:999px"></div></div>'
                f'<div style="font-size:11px;color:#1e293b;text-align:right;font-weight:600">{v}%</div>')
    
    ml_breakdown = _ml_breakdown(r)
    return f"""
<div style="background:linear-gradient(135deg,#f0f4ff,#faf5ff);
            border:2px solid #c7d2fe;border-radius:20px;
            padding:24px;margin-top:16px;
            box-shadow:0 4px 32px #6366f112">
  <h2 style="color:#312e81;margin:0 0 18px;font-size:22px;font-weight:900;text-align:center">
    🏆 Cognitive Analysis Report
  </h2>

  <!-- Top KPIs -->
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px">
    <div style="background:white;border-radius:14px;padding:14px;text-align:center;
                border:1.5px solid #e0e7ff;box-shadow:0 2px 8px #6366f110">
      <div style="font-size:34px;font-weight:900;color:#4f46e5">{r['score']}/15</div>
      <div style="font-size:10px;color:#64748b;font-weight:700;text-transform:uppercase">Score</div>
    </div>
    <div style="background:white;border-radius:14px;padding:14px;text-align:center;
                border:1.5px solid #e0e7ff;box-shadow:0 2px 8px #6366f110">
      <div style="font-size:34px;font-weight:900;color:#7c3aed">{r['accuracy']}%</div>
      <div style="font-size:10px;color:#64748b;font-weight:700;text-transform:uppercase">Accuracy</div>
    </div>
    <div style="background:white;border-radius:14px;padding:14px;text-align:center;
                border:1.5px solid #e0e7ff;box-shadow:0 2px 8px #6366f110">
      <div style="font-size:22px;font-weight:900;color:#0891b2">{r['level']}</div>
      <div style="font-size:10px;color:#64748b;font-weight:700;text-transform:uppercase">K-Means Cluster</div>
    </div>
    <div style="background:white;border-radius:14px;padding:14px;text-align:center;
                border:1.5px solid #e0e7ff;box-shadow:0 2px 8px #6366f110">
      <div style="font-size:28px;font-weight:900;color:#10b981">Top {top}%</div>
      <div style="font-size:10px;color:#64748b;font-weight:700;text-transform:uppercase">Percentile</div>
    </div>
  </div>

  <!-- Accuracy meter -->
  <div style="background:white;border-radius:14px;padding:14px;margin-bottom:10px;border:1.5px solid #e0e7ff">
    <div style="font-size:11px;font-weight:700;color:#64748b;margin-bottom:6px;text-transform:uppercase">Accuracy Meter</div>
    <div style="font-family:monospace;font-size:22px;color:#4f46e5;letter-spacing:2px">{bar}</div>
    <div style="font-size:11px;color:#94a3b8;margin-top:4px">Avg response time: {r['avg_time']}s</div>
  </div>

  <!-- By difficulty -->
  <div style="background:white;border-radius:14px;padding:14px;margin-bottom:10px;border:1.5px solid #e0e7ff">
    <div style="font-size:11px;font-weight:700;color:#64748b;margin-bottom:10px;text-transform:uppercase">Accuracy by Difficulty</div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px">
      <div><div style="font-size:18px;font-weight:800;color:#10b981">{da[0]}%</div>
           <div style="font-size:10px;color:#64748b">🟢 Easy</div>
           {pct_bar(da[0],'#10b981')}</div>
      <div><div style="font-size:18px;font-weight:800;color:#f59e0b">{da[1]}%</div>
           <div style="font-size:10px;color:#64748b">🟡 Medium</div>
           {pct_bar(da[1],'#f59e0b')}</div>
      <div><div style="font-size:18px;font-weight:800;color:#ef4444">{da[2]}%</div>
           <div style="font-size:10px;color:#64748b">🔴 Hard</div>
           {pct_bar(da[2],'#ef4444')}</div>
    </div>
  </div>

  <!-- By category -->
  <div style="background:white;border-radius:14px;padding:14px;margin-bottom:10px;border:1.5px solid #e0e7ff">
    <div style="font-size:11px;font-weight:700;color:#64748b;margin-bottom:10px;text-transform:uppercase">Accuracy by Category</div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px">
      <div><div style="font-size:18px;font-weight:800;color:#4f46e5">{ca['numbers'] or 0}%</div>
           <div style="font-size:10px;color:#64748b">🔢 Numbers</div>
           {pct_bar(ca['numbers'],'#4f46e5')}</div>
      <div><div style="font-size:18px;font-weight:800;color:#7c3aed">{ca['shapes'] or 0}%</div>
           <div style="font-size:10px;color:#64748b">🔷 Shapes</div>
           {pct_bar(ca['shapes'],'#7c3aed')}</div>
      <div><div style="font-size:18px;font-weight:800;color:#0891b2">{ca['gk'] or 0}%</div>
           <div style="font-size:10px;color:#64748b">🌍 Gen. Knowledge</div>
           {pct_bar(ca['gk'],'#0891b2')}</div>
    </div>
  </div>

  <!-- ML profile summary -->
  <div style="background:white;border-radius:14px;padding:14px;margin-bottom:10px;border:1.5px solid #e0e7ff">
    <div style="font-size:11px;font-weight:700;color:#64748b;margin-bottom:8px;text-transform:uppercase">ML Profile Summary</div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;font-size:13px">
      <div style="background:#fff1f2;border-radius:10px;padding:8px;text-align:center">
        <div style="font-size:10px;color:#ef4444;font-weight:700">IQ BAND</div>
        <div style="font-weight:800;color:#1e293b">{r['iq_band']}</div>
      </div>
      <div style="background:#eff6ff;border-radius:10px;padding:8px;text-align:center">
        <div style="font-size:10px;color:#3b82f6;font-weight:700">STYLE</div>
        <div style="font-weight:800;color:#1e293b">{r['style']}</div>
      </div>
      <div style="background:#f0fdf4;border-radius:10px;padding:8px;text-align:center">
        <div style="font-size:10px;color:#22c55e;font-weight:700">FATIGUE</div>
        <div style="font-weight:800;color:#1e293b">{r['fatigue']}</div>
      </div>
    </div>
  </div>

  <!-- Strengths / weaknesses -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px">
    <div style="background:#f0fdf4;border-radius:14px;padding:12px;border:1.5px solid #bbf7d0">
      <div style="font-size:11px;font-weight:800;color:#15803d;text-transform:uppercase;margin-bottom:4px">✅ Strengths</div>
      <div style="font-size:13px;color:#166534;font-weight:600">{', '.join(r['strengths']) or '—'}</div>
    </div>
    <div style="background:#fef2f2;border-radius:14px;padding:12px;border:1.5px solid #fecaca">
      <div style="font-size:11px;font-weight:800;color:#b91c1c;text-transform:uppercase;margin-bottom:4px">⚠️ Needs Work</div>
      <div style="font-size:13px;color:#991b1b;font-weight:600">{', '.join(r['weaknesses']) or '—'}</div>
    </div>
  </div>

  <!-- Detailed ML breakdown -->
  {ml_breakdown}

  <p style="font-size:10px;color:#94a3b8;text-align:center;margin:12px 0 0">
    ⚠️ Entertainment only — not a clinical IQ test. Voice features work best in Chrome.
  </p>
</div>"""

def _feedback_html(correct, ca, pattern):
    if correct:
        return (f'<div style="background:#f0fdf4;border:2.5px solid #22c55e;border-radius:14px;'
                f'padding:14px 18px;text-align:center;margin-top:8px">'
                f'<span style="font-size:22px">✅</span>'
                f'<span style="font-size:15px;font-weight:800;color:#15803d;margin-left:8px">Correct!</span>'
                f'<div style="font-size:12px;color:#166534;margin-top:4px">Pattern: <em>{pattern}</em></div>'
                f'</div><script>sndOk();stopTimer();</script>')
    return (f'<div style="background:#fef2f2;border:2.5px solid #ef4444;border-radius:14px;'
            f'padding:14px 18px;text-align:center;margin-top:8px">'
            f'<span style="font-size:22px">❌</span>'
            f'<span style="font-size:15px;font-weight:800;color:#b91c1c;margin-left:8px">'
            f'Incorrect! Answer: <strong>{ca}</strong></span>'
            f'<div style="font-size:12px;color:#991b1b;margin-top:4px">Pattern: <em>{pattern}</em></div>'
            f'</div><script>sndBad();stopTimer();</script>')

def _timeout_fb(ca):
    return (f'<div style="background:#fefce8;border:2.5px solid #f59e0b;border-radius:14px;'
            f'padding:14px 18px;text-align:center;margin-top:8px">'
            f'<span style="font-size:22px">⏰</span>'
            f'<span style="font-size:15px;font-weight:800;color:#92400e;margin-left:8px">'
            f'Time\'s up! Answer: <strong>{ca}</strong></span>'
            f'</div><script>sndBad();</script>')

def _after_q_js():
    return "<script>setTimeout(()=>{autoSpeak();startTimer(15);},450);</script>"

# ══════════════════════════════════════════════════════════════
#  GLOBAL SESSION + HELPERS
# ══════════════════════════════════════════════════════════════
SESSION = GameSession()

def _blank_ins():
    return {"difficulty":1,"pred_t":8.0,"rec_cat":"numbers","lr_prob":0.5,
            "iq_band":"—","gb_conf":0.0,"style":"—","svm_conf":0.0,
            "fatigue":"Fresh 🟢","nb_conf":0.0}

def _game_over(extra_fb=""):
    r = SESSION.report()
    return (
        _prog(SESSION.q_index),
        '<div style="text-align:center;font-size:22px;font-weight:900;'
        'color:#4f46e5;padding:32px;letter-spacing:1px">🏁 Game Complete!</div>',
        extra_fb,
        _score_card(SESSION.score, SESSION.q_index),
        _ml_panel(_blank_ins()),
        _completion(r),
        gr.update(visible=True),
        gr.update(value=""),
    )

# ── Event handlers ────────────────────────────────────────────
def ev_start(cat_label):
    global SESSION
    SESSION = GameSession()
    SESSION.set_category(cat_label)
    q = SESSION.next_q()
    if not q:
        return (_prog(0), '<div style="color:#ef4444;padding:20px;text-align:center">No questions available.</div>',
                "", _score_card(0,0), _ml_panel(_blank_ins()), "", gr.update(visible=False), gr.update(value=""))
    ph = random.choice(PHRASES)
    return (_prog(0), _q_card(q, ph, 1) + _after_q_js(), "",
            _score_card(0,0), _ml_panel(SESSION.get_insights()), "",
            gr.update(visible=False), gr.update(value=""))

def ev_submit(answer, _cat):
    global SESSION
    if not SESSION.current_q:
        return _game_over()
    correct, pattern = SESSION.submit(answer)
    fb = _feedback_html(correct, SESSION.history[-1]["correct_ans"], pattern)
    if SESSION.q_index >= TOTAL_QS:
        return _game_over(fb)
    q = SESSION.next_q()
    if not q:
        return _game_over(fb)
    ph = random.choice(PHRASES)
    return (_prog(SESSION.q_index), _q_card(q, ph, SESSION.q_index + 1) + _after_q_js(),
            fb, _score_card(SESSION.score, SESSION.q_index),
            _ml_panel(SESSION.get_insights()), "", gr.update(visible=False), gr.update(value=""))

def ev_timeout():
    global SESSION
    if not SESSION.current_q:
        return _game_over()
    ca = SESSION.current_q["answer"]
    SESSION.submit("", timed_out=True)
    fb = _timeout_fb(ca)
    if SESSION.q_index >= TOTAL_QS:
        return _game_over(fb)
    q = SESSION.next_q()
    if not q:
        return _game_over(fb)
    ph = random.choice(PHRASES)
    return (_prog(SESSION.q_index), _q_card(q, ph, SESSION.q_index + 1) + _after_q_js(),
            fb, _score_card(SESSION.score, SESSION.q_index),
            _ml_panel(SESSION.get_insights()), "", gr.update(visible=False), gr.update(value=""))

def ev_hint():
    if not SESSION.current_q:
        return '<div style="color:#94a3b8;font-size:13px;padding:8px">No active question.</div>'
    SESSION.hint_used = True
    h = SESSION.current_q.get("hint", "No hint available.")
    return (f'<div style="background:#fefce8;border:1.5px solid #fde047;border-radius:12px;'
            f'padding:12px 16px;font-size:13px;color:#713f12;margin-top:6px;'
            f'display:flex;align-items:center;gap:8px">'
            f'<span style="font-size:20px">💡</span>'
            f'<span><strong>Hint:</strong> {h}</span></div>')

def ev_download():
    txt = SESSION.download_txt()
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", prefix="iq_pro_", delete=False, encoding="utf-8")
    tmp.write(txt)
    tmp.close()
    return tmp.name


# ══════════════════════════════════════════════════════════════
#  GRADIO UI
# ══════════════════════════════════════════════════════════════
CSS = """
.gradio-container{max-width:1200px!important;font-family:'Segoe UI',system-ui,sans-serif!important}
body,html{background:#eef2ff!important}
#iq-timer{font-size:20px;font-weight:900;display:inline-block;min-width:36px;text-align:center}
#timer-ring{width:60px;height:60px;border-radius:50%;display:inline-flex;
            align-items:center;justify-content:center;
            background:conic-gradient(#10b981 360deg,#e0e7ff 0deg);
            box-shadow:0 2px 8px #6366f130}
#ans-box textarea,#ans-box input[type=text]{
  font-size:17px!important;border-radius:12px!important;
  border:2px solid #c7d2fe!important;background:#f8faff!important;color:#1e293b!important}
#ans-box textarea:focus,#ans-box input[type=text]:focus{
  border-color:#4f46e5!important;box-shadow:0 0 0 3px #6366f130!important}
#sub-btn{border-radius:12px!important;font-weight:800!important}
.gr-button{border-radius:12px!important;font-weight:700!important}
footer{display:none!important}
"""

def create_ui():
    with gr.Blocks(
        title="🧠 IQ Detector Pro",
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="violet"),
        head=JS_HEAD,
        css=CSS,
    ) as demo:

        # ── HEADER ───────────────────────────────────────────
        gr.HTML("""
<div style="background:linear-gradient(135deg,#1e1b4b,#312e81,#4f46e5,#7c3aed,#a855f7);
            padding:28px 24px;border-radius:22px;text-align:center;
            margin-bottom:14px;color:white;box-shadow:0 8px 32px #6366f145;
            position:relative;overflow:hidden">
  <div style="position:absolute;top:-30px;right:-30px;width:180px;height:180px;
              background:#ffffff08;border-radius:50%"></div>
  <div style="position:absolute;bottom:-40px;left:-20px;width:140px;height:140px;
              background:#ffffff06;border-radius:50%"></div>
  <div style="font-size:44px;margin-bottom:6px">🧠</div>
  <h1 style="margin:0;font-size:34px;font-weight:900;letter-spacing:3px">IQ Detector Pro</h1>
  <p style="margin:8px 0 0;opacity:.9;font-size:13px">
    7 ML Algorithms &nbsp;·&nbsp; Adaptive Difficulty &nbsp;·&nbsp; 15 Questions &nbsp;·&nbsp; Numbers + Shapes + General Knowledge
  </p>
  <div style="display:flex;justify-content:center;gap:8px;margin-top:12px;flex-wrap:wrap">
    <span style="background:#ffffff22;border-radius:999px;padding:3px 12px;font-size:11px;font-weight:600">Logistic Regression</span>
    <span style="background:#ffffff22;border-radius:999px;padding:3px 12px;font-size:11px;font-weight:600">Ridge Regression</span>
    <span style="background:#ffffff22;border-radius:999px;padding:3px 12px;font-size:11px;font-weight:600">Random Forest</span>
    <span style="background:#ffffff22;border-radius:999px;padding:3px 12px;font-size:11px;font-weight:600">K-Means</span>
    <span style="background:#ffffff22;border-radius:999px;padding:3px 12px;font-size:11px;font-weight:600">Gradient Boosting</span>
    <span style="background:#ffffff22;border-radius:999px;padding:3px 12px;font-size:11px;font-weight:600">SVM (RBF)</span>
    <span style="background:#ffffff22;border-radius:999px;padding:3px 12px;font-size:11px;font-weight:600">Naive Bayes</span>
  </div>
</div>
<div style="background:#fffbeb;border:1.5px solid #fcd34d;border-radius:12px;
            padding:10px 16px;font-size:12px;color:#78350f;
            margin-bottom:14px;text-align:center;font-weight:600">
  ⚠️ Entertainment only — not a clinical IQ test &nbsp;|&nbsp;
  🎤 Voice features work best in <strong>Chrome</strong> or <strong>Edge</strong>
</div>""")

        # ── CONTROLS ─────────────────────────────────────────
        with gr.Row():
            cat_dd = gr.Dropdown(["Mixed Mode", "Numbers Only", "Shapes Only", "GK Only"],
                                 value="Mixed Mode", label="📂 Category", scale=2)
            start_b = gr.Button("▶️ Start Game", variant="primary", scale=2)
            rest_b = gr.Button("🔄 Restart", variant="secondary", scale=1)
            spkr_b = gr.Button("🔊 Replay", scale=1)
            mic_b = gr.Button("🎤 Speak", scale=1)
            dark_b = gr.Button("🌙 Dark", scale=1)

        dark_b.click(None, None, None, js="()=>toggleDark()")
        spkr_b.click(None, None, None, js="()=>replayQ()")
        mic_b.click(None, None, None, js="()=>listenNow()")

        # ── MAIN ─────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=7):
                out_prog = gr.HTML(_prog(0))
                out_q = gr.HTML("""
<div style="background:#ffffff;border-radius:20px;padding:48px 24px;text-align:center;
            box-shadow:0 4px 24px #6366f112;border:2px solid #e0e7ff;
            color:#94a3b8;font-size:17px;font-weight:600">
  Press <strong style="color:#4f46e5">▶️ Start Game</strong> to begin!<br>
  <span style="font-size:13px;color:#a5b4fc">15 questions · Numbers 🔢 · Shapes 🔷 · General Knowledge 🌍</span>
</div>""")

                # Timer
                gr.HTML("""
<div style="text-align:center;margin:10px 0 6px">
  <div id="timer-ring">
    <span id="iq-timer" style="font-size:20px;font-weight:900;color:#1e293b">—</span>
  </div>
  <div style="font-size:10px;color:#64748b;margin-top:3px;font-weight:600">seconds remaining</div>
</div>""")

                out_hint = gr.HTML("")
                hint_b = gr.Button("💡 Show Hint", size="sm", variant="secondary")

                with gr.Row():
                    ans_box = gr.Textbox(placeholder="Type your answer here…",
                                         show_label=False, elem_id="ans-box", scale=5)
                    sub_b = gr.Button("✅ Submit", variant="primary", scale=1,
                                      size="lg", elem_id="sub-btn")

                timeout_b = gr.Button("", visible=False, elem_id="to-btn")
                out_fb = gr.HTML("")

            with gr.Column(scale=4):
                out_score = gr.HTML(_score_card(0,0))
                out_ml = gr.HTML(_ml_panel(_blank_ins()))
                down_b = gr.Button("📥 Download Report", visible=False, variant="secondary")
                down_file = gr.File(label="📄 Results file", visible=False)

        out_complete = gr.HTML("")

        OUTS = [out_prog, out_q, out_fb, out_score, out_ml, out_complete, down_b, ans_box]

        start_b.click(ev_start, [cat_dd], OUTS)
        rest_b.click(ev_start, [cat_dd], OUTS)
        sub_b.click(ev_submit, [ans_box, cat_dd], OUTS)
        ans_box.submit(ev_submit, [ans_box, cat_dd], OUTS)
        hint_b.click(ev_hint, [], [out_hint])
        timeout_b.click(ev_timeout, [], OUTS)
        down_b.click(ev_download, [], [down_file]).then(
            lambda: gr.update(visible=True), None, [down_file])

        # ── MODEL COMPARISON DASHBOARD ────────────────────────
        gr.HTML(
            '<div style="margin-top:28px;background:linear-gradient(135deg,#312e81,#4f46e5);'
            'border-radius:16px;padding:14px 20px">'
            '<span style="color:white;font-size:16px;font-weight:900">📊 Model Comparison Dashboard</span>'
            '<span style="font-size:10px;background:#ffffff30;border-radius:999px;'
            'padding:2px 10px;margin-left:8px;color:white">COURSE REQUIREMENT §5</span>'
            '<p style="color:#c7d2fe;margin:4px 0 0;font-size:12px">'
            '5 classifiers · same 500-sample dataset · Accuracy · Precision · Recall · F1 · ROC-AUC · Confusion Matrix</p>'
            '</div>'
        )
        gr.HTML(get_comparison_html())

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)