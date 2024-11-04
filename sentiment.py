{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "df24d4b1",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "plt.style.use('ggplot')\n",
    "\n",
    "import nltk"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "b74c5d92",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(993, 1)\n"
     ]
    }
   ],
   "source": [
    "df = pd.read_csv('clean_headlines.csv')\n",
    "print(df.shape)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "f716dc39",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>that's my UAlberta :')</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>I got the last croissant.</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Every upper level Student in their 100 level o...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>400 upvotes and we will put a cowboy hat on th...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Hmmm</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                               title\n",
       "0                             that's my UAlberta :')\n",
       "1                          I got the last croissant.\n",
       "2  Every upper level Student in their 100 level o...\n",
       "3  400 upvotes and we will put a cowboy hat on th...\n",
       "4                                               Hmmm"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c735f835",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "a1950bfa",
   "metadata": {},
   "outputs": [],
   "source": [
    "headlines = df['title']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "8b42d2de",
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "expected string or bytes-like object, got 'Series'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[38], line 1\u001b[0m\n\u001b[0;32m----> 1\u001b[0m tokens \u001b[38;5;241m=\u001b[39m nltk\u001b[38;5;241m.\u001b[39mword_tokenize(headlines)\n\u001b[1;32m      2\u001b[0m token[:\u001b[38;5;241m10\u001b[39m]\n",
      "File \u001b[0;32m~/anaconda3/lib/python3.11/site-packages/nltk/tokenize/__init__.py:129\u001b[0m, in \u001b[0;36mword_tokenize\u001b[0;34m(text, language, preserve_line)\u001b[0m\n\u001b[1;32m    114\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mword_tokenize\u001b[39m(text, language\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124menglish\u001b[39m\u001b[38;5;124m\"\u001b[39m, preserve_line\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mFalse\u001b[39;00m):\n\u001b[1;32m    115\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[1;32m    116\u001b[0m \u001b[38;5;124;03m    Return a tokenized copy of *text*,\u001b[39;00m\n\u001b[1;32m    117\u001b[0m \u001b[38;5;124;03m    using NLTK's recommended word tokenizer\u001b[39;00m\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m    127\u001b[0m \u001b[38;5;124;03m    :type preserve_line: bool\u001b[39;00m\n\u001b[1;32m    128\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[0;32m--> 129\u001b[0m     sentences \u001b[38;5;241m=\u001b[39m [text] \u001b[38;5;28;01mif\u001b[39;00m preserve_line \u001b[38;5;28;01melse\u001b[39;00m sent_tokenize(text, language)\n\u001b[1;32m    130\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m [\n\u001b[1;32m    131\u001b[0m         token \u001b[38;5;28;01mfor\u001b[39;00m sent \u001b[38;5;129;01min\u001b[39;00m sentences \u001b[38;5;28;01mfor\u001b[39;00m token \u001b[38;5;129;01min\u001b[39;00m _treebank_word_tokenizer\u001b[38;5;241m.\u001b[39mtokenize(sent)\n\u001b[1;32m    132\u001b[0m     ]\n",
      "File \u001b[0;32m~/anaconda3/lib/python3.11/site-packages/nltk/tokenize/__init__.py:107\u001b[0m, in \u001b[0;36msent_tokenize\u001b[0;34m(text, language)\u001b[0m\n\u001b[1;32m     97\u001b[0m \u001b[38;5;250m\u001b[39m\u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[1;32m     98\u001b[0m \u001b[38;5;124;03mReturn a sentence-tokenized copy of *text*,\u001b[39;00m\n\u001b[1;32m     99\u001b[0m \u001b[38;5;124;03musing NLTK's recommended sentence tokenizer\u001b[39;00m\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m    104\u001b[0m \u001b[38;5;124;03m:param language: the model name in the Punkt corpus\u001b[39;00m\n\u001b[1;32m    105\u001b[0m \u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[1;32m    106\u001b[0m tokenizer \u001b[38;5;241m=\u001b[39m load(\u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mtokenizers/punkt/\u001b[39m\u001b[38;5;132;01m{\u001b[39;00mlanguage\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m.pickle\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m--> 107\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m tokenizer\u001b[38;5;241m.\u001b[39mtokenize(text)\n",
      "File \u001b[0;32m~/anaconda3/lib/python3.11/site-packages/nltk/tokenize/punkt.py:1281\u001b[0m, in \u001b[0;36mPunktSentenceTokenizer.tokenize\u001b[0;34m(self, text, realign_boundaries)\u001b[0m\n\u001b[1;32m   1277\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mtokenize\u001b[39m(\u001b[38;5;28mself\u001b[39m, text: \u001b[38;5;28mstr\u001b[39m, realign_boundaries: \u001b[38;5;28mbool\u001b[39m \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mTrue\u001b[39;00m) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m List[\u001b[38;5;28mstr\u001b[39m]:\n\u001b[1;32m   1278\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[1;32m   1279\u001b[0m \u001b[38;5;124;03m    Given a text, returns a list of the sentences in that text.\u001b[39;00m\n\u001b[1;32m   1280\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[0;32m-> 1281\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mlist\u001b[39m(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39msentences_from_text(text, realign_boundaries))\n",
      "File \u001b[0;32m~/anaconda3/lib/python3.11/site-packages/nltk/tokenize/punkt.py:1341\u001b[0m, in \u001b[0;36mPunktSentenceTokenizer.sentences_from_text\u001b[0;34m(self, text, realign_boundaries)\u001b[0m\n\u001b[1;32m   1332\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21msentences_from_text\u001b[39m(\n\u001b[1;32m   1333\u001b[0m     \u001b[38;5;28mself\u001b[39m, text: \u001b[38;5;28mstr\u001b[39m, realign_boundaries: \u001b[38;5;28mbool\u001b[39m \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mTrue\u001b[39;00m\n\u001b[1;32m   1334\u001b[0m ) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m List[\u001b[38;5;28mstr\u001b[39m]:\n\u001b[1;32m   1335\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[1;32m   1336\u001b[0m \u001b[38;5;124;03m    Given a text, generates the sentences in that text by only\u001b[39;00m\n\u001b[1;32m   1337\u001b[0m \u001b[38;5;124;03m    testing candidate sentence breaks. If realign_boundaries is\u001b[39;00m\n\u001b[1;32m   1338\u001b[0m \u001b[38;5;124;03m    True, includes in the sentence closing punctuation that\u001b[39;00m\n\u001b[1;32m   1339\u001b[0m \u001b[38;5;124;03m    follows the period.\u001b[39;00m\n\u001b[1;32m   1340\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[0;32m-> 1341\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m [text[s:e] \u001b[38;5;28;01mfor\u001b[39;00m s, e \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mspan_tokenize(text, realign_boundaries)]\n",
      "File \u001b[0;32m~/anaconda3/lib/python3.11/site-packages/nltk/tokenize/punkt.py:1341\u001b[0m, in \u001b[0;36m<listcomp>\u001b[0;34m(.0)\u001b[0m\n\u001b[1;32m   1332\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21msentences_from_text\u001b[39m(\n\u001b[1;32m   1333\u001b[0m     \u001b[38;5;28mself\u001b[39m, text: \u001b[38;5;28mstr\u001b[39m, realign_boundaries: \u001b[38;5;28mbool\u001b[39m \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mTrue\u001b[39;00m\n\u001b[1;32m   1334\u001b[0m ) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m List[\u001b[38;5;28mstr\u001b[39m]:\n\u001b[1;32m   1335\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[1;32m   1336\u001b[0m \u001b[38;5;124;03m    Given a text, generates the sentences in that text by only\u001b[39;00m\n\u001b[1;32m   1337\u001b[0m \u001b[38;5;124;03m    testing candidate sentence breaks. If realign_boundaries is\u001b[39;00m\n\u001b[1;32m   1338\u001b[0m \u001b[38;5;124;03m    True, includes in the sentence closing punctuation that\u001b[39;00m\n\u001b[1;32m   1339\u001b[0m \u001b[38;5;124;03m    follows the period.\u001b[39;00m\n\u001b[1;32m   1340\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[0;32m-> 1341\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m [text[s:e] \u001b[38;5;28;01mfor\u001b[39;00m s, e \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mspan_tokenize(text, realign_boundaries)]\n",
      "File \u001b[0;32m~/anaconda3/lib/python3.11/site-packages/nltk/tokenize/punkt.py:1329\u001b[0m, in \u001b[0;36mPunktSentenceTokenizer.span_tokenize\u001b[0;34m(self, text, realign_boundaries)\u001b[0m\n\u001b[1;32m   1327\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m realign_boundaries:\n\u001b[1;32m   1328\u001b[0m     slices \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_realign_boundaries(text, slices)\n\u001b[0;32m-> 1329\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m sentence \u001b[38;5;129;01min\u001b[39;00m slices:\n\u001b[1;32m   1330\u001b[0m     \u001b[38;5;28;01myield\u001b[39;00m (sentence\u001b[38;5;241m.\u001b[39mstart, sentence\u001b[38;5;241m.\u001b[39mstop)\n",
      "File \u001b[0;32m~/anaconda3/lib/python3.11/site-packages/nltk/tokenize/punkt.py:1459\u001b[0m, in \u001b[0;36mPunktSentenceTokenizer._realign_boundaries\u001b[0;34m(self, text, slices)\u001b[0m\n\u001b[1;32m   1446\u001b[0m \u001b[38;5;250m\u001b[39m\u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[1;32m   1447\u001b[0m \u001b[38;5;124;03mAttempts to realign punctuation that falls after the period but\u001b[39;00m\n\u001b[1;32m   1448\u001b[0m \u001b[38;5;124;03mshould otherwise be included in the same sentence.\u001b[39;00m\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m   1456\u001b[0m \u001b[38;5;124;03m    [\"(Sent1.)\", \"Sent2.\"].\u001b[39;00m\n\u001b[1;32m   1457\u001b[0m \u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[1;32m   1458\u001b[0m realign \u001b[38;5;241m=\u001b[39m \u001b[38;5;241m0\u001b[39m\n\u001b[0;32m-> 1459\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m sentence1, sentence2 \u001b[38;5;129;01min\u001b[39;00m _pair_iter(slices):\n\u001b[1;32m   1460\u001b[0m     sentence1 \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mslice\u001b[39m(sentence1\u001b[38;5;241m.\u001b[39mstart \u001b[38;5;241m+\u001b[39m realign, sentence1\u001b[38;5;241m.\u001b[39mstop)\n\u001b[1;32m   1461\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m sentence2:\n",
      "File \u001b[0;32m~/anaconda3/lib/python3.11/site-packages/nltk/tokenize/punkt.py:321\u001b[0m, in \u001b[0;36m_pair_iter\u001b[0;34m(iterator)\u001b[0m\n\u001b[1;32m    319\u001b[0m iterator \u001b[38;5;241m=\u001b[39m \u001b[38;5;28miter\u001b[39m(iterator)\n\u001b[1;32m    320\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[0;32m--> 321\u001b[0m     prev \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mnext\u001b[39m(iterator)\n\u001b[1;32m    322\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mStopIteration\u001b[39;00m:\n\u001b[1;32m    323\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m\n",
      "File \u001b[0;32m~/anaconda3/lib/python3.11/site-packages/nltk/tokenize/punkt.py:1431\u001b[0m, in \u001b[0;36mPunktSentenceTokenizer._slices_from_text\u001b[0;34m(self, text)\u001b[0m\n\u001b[1;32m   1429\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m_slices_from_text\u001b[39m(\u001b[38;5;28mself\u001b[39m, text: \u001b[38;5;28mstr\u001b[39m) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m Iterator[\u001b[38;5;28mslice\u001b[39m]:\n\u001b[1;32m   1430\u001b[0m     last_break \u001b[38;5;241m=\u001b[39m \u001b[38;5;241m0\u001b[39m\n\u001b[0;32m-> 1431\u001b[0m     \u001b[38;5;28;01mfor\u001b[39;00m match, context \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_match_potential_end_contexts(text):\n\u001b[1;32m   1432\u001b[0m         \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mtext_contains_sentbreak(context):\n\u001b[1;32m   1433\u001b[0m             \u001b[38;5;28;01myield\u001b[39;00m \u001b[38;5;28mslice\u001b[39m(last_break, match\u001b[38;5;241m.\u001b[39mend())\n",
      "File \u001b[0;32m~/anaconda3/lib/python3.11/site-packages/nltk/tokenize/punkt.py:1395\u001b[0m, in \u001b[0;36mPunktSentenceTokenizer._match_potential_end_contexts\u001b[0;34m(self, text)\u001b[0m\n\u001b[1;32m   1393\u001b[0m previous_slice \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mslice\u001b[39m(\u001b[38;5;241m0\u001b[39m, \u001b[38;5;241m0\u001b[39m)\n\u001b[1;32m   1394\u001b[0m previous_match \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[0;32m-> 1395\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m match \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_lang_vars\u001b[38;5;241m.\u001b[39mperiod_context_re()\u001b[38;5;241m.\u001b[39mfinditer(text):\n\u001b[1;32m   1396\u001b[0m \n\u001b[1;32m   1397\u001b[0m     \u001b[38;5;66;03m# Get the slice of the previous word\u001b[39;00m\n\u001b[1;32m   1398\u001b[0m     before_text \u001b[38;5;241m=\u001b[39m text[previous_slice\u001b[38;5;241m.\u001b[39mstop : match\u001b[38;5;241m.\u001b[39mstart()]\n\u001b[1;32m   1399\u001b[0m     index_after_last_space \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_get_last_whitespace_index(before_text)\n",
      "\u001b[0;31mTypeError\u001b[0m: expected string or bytes-like object, got 'Series'"
     ]
    }
   ],
   "source": [
    "tokens = nltk.word_tokenize(headlines)\n",
    "token[:10]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6d0310dc",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5ace8614",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6df0d282",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c81f19cf",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
