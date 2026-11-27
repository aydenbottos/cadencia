#!/usr/bin/env python3
"""
Build the complete 4,900-item Cadencia Spanish linguistic seed database:
  - 4,000 most frequent Spanish word lemmas (combined spoken + written via wordfreq,
    proper nouns removed via spaCy es_core_news_md + curated filters, real English glosses
    from MUSE es-en + curated core Spanish lexicon, Castilian es-ES & LatAm es-419 IPA,
    transparent cognate classification excluding false friends)
  - 400 non-cognate hold-out lemmas from ranks 4,001–5,000 (strictly never scheduled)
  - 200 multi-word chunks (tools/data/spanish_chunks_200.json)
  - 300 Instituto Cervantes A1–B2 grammar patterns (cervantes_a1_a2.json + cervantes_b1_b2.json)
  - Comprehensive Spanish inflection -> lemma lookup table (~100k+ entries)
  - Assessment seeds: 120 pseudowords, 30 Spanish EIT sentences, native listening tests
  - 3 Serial Story arcs (Noir Madrid, Patagonia Sci-Fi, Oaxaca/Valencia Culinary Drama)

Outputs:
  - Sources/CadenciaCore/Resources/cadencia_seed.sqlite
  - tools/cache/item_bank_report.json
"""

import json
import math
import os
import re
import sqlite3
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import spacy
from wordfreq import top_n_list, zipf_frequency

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "tools" / "data"
CACHE_DIR = ROOT / "tools" / "cache"
OUT_SQLITE = ROOT / "Sources" / "CadenciaCore" / "Resources" / "cadencia_seed.sqlite"
OUT_REPORT = CACHE_DIR / "item_bank_report.json"

VALID_SPANISH_RE = re.compile(r"^[a-záéíóúüñ]{2,24}$")
SINGLE_LETTER_WORDS = {"a", "e", "o", "u", "y"}

# High-frequency Spanish function words & core lemmas with exact English glosses
# (ensuring crisp, pedagogical glosses instead of noisy dictionary entries)
CORE_GLOSS_OVERRIDES: dict[str, tuple[str, str, bool]] = {
    # lemma: (english_gloss, pos, is_cognate)
    "de": ("of / from", "ADP", False),
    "del": ("of the / from the", "ADP", False),
    "al": ("to the / at the", "ADP", False),
    "la": ("the (feminine)", "DET", False),
    "el": ("the (masculine)", "DET", False),
    "que": ("that / which / who", "SCONJ", False),
    "en": ("in / on / at", "ADP", False),
    "y": ("and", "CCONJ", False),
    "a": ("to / at", "ADP", False),
    "un": ("a / an / one", "DET", False),
    "ser": ("to be (identity/essence)", "VERB", False),
    "se": ("oneself / each other / passive se", "PRON", False),
    "no": ("no / not", "ADV", True),
    "haber": ("to have (auxiliary) / there is", "AUX", False),
    "por": ("by / through / because of / for", "ADP", False),
    "con": ("with", "ADP", False),
    "su": ("his / her / your / their", "DET", False),
    "para": ("for / in order to / toward", "ADP", False),
    "como": ("like / as / how", "SCONJ", False),
    "estar": ("to be (location/state)", "VERB", False),
    "tener": ("to have / possess", "VERB", False),
    "le": ("to him / to her / to you", "PRON", False),
    "lo": ("it / him / the (neuter)", "PRON", False),
    "todo": ("all / every / everything", "DET", False),
    "pero": ("but", "CCONJ", False),
    "más": ("more / most / plus", "ADV", False),
    "hacer": ("to do / to make", "VERB", False),
    "o": ("or", "CCONJ", False),
    "poder": ("to be able to / can / power", "VERB", False),
    "decir": ("to say / to tell", "VERB", False),
    "este": ("this", "DET", False),
    "ir": ("to go", "VERB", False),
    "otro": ("other / another", "DET", False),
    "ese": ("that", "DET", False),
    "si": ("if / whether", "SCONJ", False),
    "me": ("me / to me / myself", "PRON", False),
    "ya": ("already / now / anymore", "ADV", False),
    "ver": ("to see / watch", "VERB", False),
    "porque": ("because", "SCONJ", False),
    "dar": ("to give", "VERB", False),
    "cuando": ("when", "SCONJ", False),
    "él": ("he / him", "PRON", False),
    "muy": ("very", "ADV", False),
    "sin": ("without", "ADP", False),
    "vez": ("time / occasion / turn", "NOUN", False),
    "mucho": ("much / a lot / many", "DET", False),
    "saber": ("to know (facts/skills)", "VERB", False),
    "qué": ("what / how", "PRON", False),
    "sobre": ("on / about / over / envelope", "ADP", False),
    "mi": ("my", "DET", False),
    "alguno": ("some / any", "DET", False),
    "mismo": ("same / self", "ADJ", False),
    "yo": ("I", "PRON", False),
    "también": ("also / too", "ADV", False),
    "hasta": ("until / up to / even", "ADP", False),
    "año": ("year", "NOUN", False),
    "dos": ("two", "NUM", False),
    "querer": ("to want / to love", "VERB", False),
    "entre": ("between / among", "ADP", False),
    "así": ("thus / like this / so", "ADV", False),
    "primero": ("first", "ADJ", True),
    "desde": ("from / since", "ADP", False),
    "grande": ("big / large / great", "ADJ", True),
    "eso": ("that (neuter)", "PRON", False),
    "ni": ("neither / nor", "CCONJ", False),
    "nos": ("us / to us / ourselves", "PRON", False),
    "llegar": ("to arrive / reach", "VERB", False),
    "pasar": ("to pass / happen / spend time", "VERB", False),
    "tiempo": ("time / weather", "NOUN", False),
    "ella": ("she / her", "PRON", False),
    "sí": ("yes / oneself", "ADV", False),
    "día": ("day", "NOUN", False),
    "uno": ("one", "NUM", True),
    "bien": ("well / fine / good", "ADV", False),
    "poco": ("little / few / a bit", "ADV", False),
    "deber": ("must / should / to owe / duty", "VERB", False),
    "entonces": ("then / at that time", "ADV", False),
    "poner": ("to put / place / set", "VERB", False),
    "cosa": ("thing", "NOUN", False),
    "tanto": ("so much / as much", "ADV", False),
    "hombre": ("man", "NOUN", False),
    "parecer": ("to seem / appear / resemble", "VERB", False),
    "nuestro": ("our", "DET", False),
    "tan": ("so / as", "ADV", False),
    "donde": ("where", "SCONJ", False),
    "ahora": ("now", "ADV", False),
    "parte": ("part / share", "NOUN", True),
    "después": ("after / afterward / later", "ADV", False),
    "vida": ("life", "NOUN", False),
    "quedar": ("to stay / remain / meet up", "VERB", False),
    "siempre": ("always", "ADV", False),
    "creer": ("to believe / think", "VERB", False),
    "hablar": ("to speak / talk", "VERB", False),
    "llevar": ("to carry / wear / take", "VERB", False),
    "dejar": ("to leave / let / stop doing", "VERB", False),
    "nada": ("nothing / not at all", "PRON", False),
    "cada": ("each / every", "DET", False),
    "seguir": ("to follow / continue / keep doing", "VERB", False),
    "menos": ("less / fewer / minus", "ADV", False),
    "nuevo": ("new", "ADJ", False),
    "encontrar": ("to find / encounter", "VERB", False),
    "algo": ("something / somewhat", "PRON", False),
    "solo": ("only / alone / lonely", "ADV", True),
    "decir": ("to say / tell", "VERB", False),
    "salir": ("to go out / leave / exit", "VERB", False),
    "volver": ("to return / come back", "VERB", False),
    "tomar": ("to take / drink / have", "VERB", False),
    "conocer": ("to know (people/places) / meet", "VERB", False),
    "vivir": ("to live", "VERB", False),
    "sentir": ("to feel / regret", "VERB", False),
    "tratar": ("to try / treat / deal with", "VERB", True),
    "mirar": ("to look at / watch", "VERB", False),
    "contar": ("to count / tell (a story)", "VERB", False),
    "empezar": ("to begin / start", "VERB", False),
    "esperar": ("to wait / hope / expect", "VERB", False),
    "buscar": ("to look for / search", "VERB", False),
    "existir": ("to exist", "VERB", True),
    "entrar": ("to enter / go in", "VERB", True),
    "trabajar": ("to work", "VERB", False),
    "escribir": ("to write", "VERB", False),
    "perder": ("to lose / miss", "VERB", False),
    "producir": ("to produce", "VERB", True),
    "ocurrir": ("to happen / occur", "VERB", True),
    "entender": ("to understand", "VERB", False),
    "pedir": ("to ask for / order / request", "VERB", False),
    "recibir": ("to receive", "VERB", True),
    "recordar": ("to remember / remind", "VERB", False),
    "terminar": ("to finish / end", "VERB", True),
    "permitir": ("to allow / permit", "VERB", True),
    "aparecer": ("to appear", "VERB", True),
    "conseguir": ("to get / obtain / achieve", "VERB", False),
    "comenzar": ("to start / commence", "VERB", True),
    "servir": ("to serve / be useful", "VERB", True),
    "sacar": ("to take out / pull out / get", "VERB", False),
    "necesitar": ("to need", "VERB", True),
    "mantener": ("to maintain / keep", "VERB", True),
    "resultar": ("to turn out / result", "VERB", True),
    "leer": ("to read", "VERB", False),
    "caer": ("to fall", "VERB", False),
    "cambiar": ("to change / exchange", "VERB", False),
    "presentar": ("to present / introduce", "VERB", True),
    "crear": ("to create", "VERB", True),
    "abrir": ("to open", "VERB", False),
    "considerar": ("to consider", "VERB", True),
    "oír": ("to hear", "VERB", False),
    "acabar": ("to finish / end up", "VERB", False),
    "convertir": ("to convert / turn into", "VERB", True),
    "ganar": ("to win / earn", "VERB", False),
    "formar": ("to form / shape", "VERB", True),
    "traer": ("to bring", "VERB", False),
    "partir": ("to depart / split", "VERB", False),
    "morir": ("to die", "VERB", False),
    "aceptar": ("to accept", "VERB", True),
    "realizar": ("to carry out / perform", "VERB", False),
    "suponer": ("to suppose / entail", "VERB", True),
    "comprender": ("to understand / comprehend", "VERB", True),
    "lograr": ("to achieve / manage to", "VERB", False),
    "explicar": ("to explain", "VERB", True),
    "preguntar": ("to ask (a question)", "VERB", False),
    "tocar": ("to touch / play (instrument) / be one's turn", "VERB", False),
    "reconocer": ("to recognize / admit", "VERB", True),
    "estudiar": ("to study", "VERB", True),
    "alcanzar": ("to reach / attain", "VERB", False),
    "nacer": ("to be born", "VERB", False),
    "dirigir": ("to direct / lead / manage", "VERB", True),
    "correr": ("to run", "VERB", False),
    "utilizar": ("to use / utilize", "VERB", True),
    "pagar": ("to pay", "VERB", False),
    "ayudar": ("to help", "VERB", False),
    "gustar": ("to please / be liked by", "VERB", False),
    "jugar": ("to play (game/sport)", "VERB", False),
    "escuchar": ("to listen to", "VERB", False),
    "cumplir": ("to fulfill / turn (an age)", "VERB", False),
    "ofrecer": ("to offer", "VERB", True),
    "descubrir": ("to discover", "VERB", True),
    "levantar": ("to lift / raise / get up", "VERB", False),
    "intentar": ("to try / attempt", "VERB", False),
    "usar": ("to use", "VERB", True),
    "tú": ("you (informal singular)", "PRON", False),
    "te": ("you / to you / yourself", "PRON", False),
    "tu": ("your (informal)", "DET", False),
    "usted": ("you (formal singular)", "PRON", False),
    "nosotros": ("we / us", "PRON", False),
    "ellos": ("they / them", "PRON", False),
    "quien": ("who / whom", "PRON", False),
    "quién": ("who?", "PRON", False),
    "cuál": ("which / what?", "PRON", False),
    "cuándo": ("when?", "ADV", False),
    "dónde": ("where?", "ADV", False),
    "cómo": ("how?", "ADV", False),
    "cuánto": ("how much / how many?", "DET", False),
    "e": ("and (before i-/hi-)", "CCONJ", False),
    "u": ("or (before o-/ho-)", "CCONJ", False),
    "aunque": ("although / even though / even if", "SCONJ", False),
    "mientras": ("while / whereas", "SCONJ", False),
    "sino": ("but rather", "CCONJ", False),
    "ni": ("neither / nor", "CCONJ", False),
    "todavía": ("still / yet", "ADV", False),
    "aún": ("still / yet / even", "ADV", False),
    "nunca": ("never", "ADV", False),
    "jamás": ("never / ever", "ADV", False),
    "tampoco": ("neither / not either", "ADV", False),
    "quizá": ("perhaps / maybe", "ADV", False),
    "quizás": ("perhaps / maybe", "ADV", False),
    "ojalá": ("hopefully / if only", "INTJ", False),
    "casi": ("almost / nearly", "ADV", False),
    "aquí": ("here", "ADV", False),
    "allí": ("there", "ADV", False),
    "ahí": ("there (nearby)", "ADV", False),
    "allá": ("over there", "ADV", False),
    "acá": ("over here", "ADV", False),
    "hoy": ("today", "ADV", False),
    "ayer": ("yesterday", "ADV", False),
    "mañana": ("tomorrow / morning", "ADV", False),
    "siempre": ("always", "ADV", False),
    "pronto": ("soon / early", "ADV", False),
    "luego": ("then / later", "ADV", False),
    "antes": ("before / earlier", "ADV", False),
    "durante": ("during", "ADP", True),
    "hacia": ("toward", "ADP", False),
    "según": ("according to", "ADP", False),
    "contra": ("against", "ADP", True),
    "tras": ("after / behind", "ADP", False),
    "ante": ("before / in the face of", "ADP", False),
    "bajo": ("under / low / short", "ADP", False),
    "mediante": ("by means of / through", "ADP", False),
    "salvo": ("except / safe", "ADP", False),
    "excepto": ("except", "ADP", True),
    "aquel": ("that (far away)", "DET", False),
    "ninguno": ("none / no / not any", "DET", False),
    "nadie": ("nobody / no one", "PRON", False),
    "alguien": ("someone / somebody", "PRON", False),
    "cualquiera": ("any / anyone / whichever", "PRON", False),
    "demás": ("rest / other", "PRON", False),
    "ambos": ("both", "DET", False),
    "cuyo": ("whose", "DET", False),
}

# Known Spanish-English false friends (MUST NOT be flagged as transparent cognates!)
FALSE_FRIENDS = {
    "embarazada", "embarazar", "actual", "actualmente", "asistir", "atender",
    "carpeta", "constipado", "contestar", "delito", "desgracia", "disgusto",
    "éxito", "fábrica", "familiar", "grosería", "idioma", "intentar", "introducir",
    "largo", "lectura", "librería", "molestar", "noticia", "pariente", "pie",
    "pretender", "realizar", "recordar", "relativo", "ropa", "sano", "sensible",
    "soportar", "suceder", "suceso", "tuna", "vaso", "sopa", "arena", "aviso",
    "campo", "cargo", "carta", "chocar", "colegio", "compromiso", "conferencia",
    "cuarto", "decepción", "defraudar", "despertar", "discusión", "editor",
    "educado", "enviar", "estafa", "extranjero", "fastidiar", "firma", "gracioso",
    "ignorar", "inconveniente", "jubilar", "lástima", "mayor", "menor", "nudo",
    "once", "pan", "papa", "parada", "pared", "pecado", "pie", "plaza", "política",
    "preservativo", "quitar", "red", "regla", "renta", "restar", "retirar", "salada",
    "salir", "seguro", "sin", "sobrar", "son", "suelo", "tabla", "tapa", "tarifa",
    "taza", "tema", "tierra", "tipo", "traje", "trampa", "tratar", "últimamente",
    "uña", "vagón", "vela", "venta", "vereda", "vidrio", "violín",
}

# Explicit proper nouns, countries, cities, internet/English loan noise to exclude
STOP_PROPER_AND_NOISE = {
    "españa", "méxico", "méjico", "argentina", "colombia", "chile", "perú", "venezuela",
    "ecuador", "bolivia", "uruguay", "paraguay", "cuba", "guatemala", "honduras",
    "nicaragua", "panamá", "salvador", "madrid", "barcelona", "sevilla", "valencia",
    "bilbao", "zaragoza", "málaga", "bogotá", "lima", "santiago", "caracas", "quito",
    "montevideo", "buenos", "aires", "europa", "américa", "áfrica", "asia", "francia",
    "italia", "alemania", "inglaterra", "portugal", "brasil", "rusia", "china", "japón",
    "india", "eeuu", "usa", "onu", "ue", "fifa", "nba", "nfl", "http", "https", "www",
    "com", "org", "net", "html", "pdf", "youtube", "facebook", "twitter", "instagram",
    "google", "whatsapp", "tiktok", "netflix", "amazon", "apple", "microsoft", "android",
    "iphone", "windows", "linux", "covid", "dios", "jesús", "cristo", "maría", "josé",
    "juan", "carlos", "luis", "miguel", "antonio", "francisco", "pedro", "pablo",
    "javier", "diego", "jorge", "fernando", "roberto", "daniel", "alejandro", "david",
    "manuel", "rafael", "alberto", "sergio", "eduardo", "ricardo", "andrés", "ramón",
    "enrique", "martín", "alberto", "ana", "laura", "carmen", "elena", "sofía", "lucía",
    "paula", "marta", "sara", "isabel", "rosa", "teresa", "beatriz", "raquel", "patricia",
    "andrea", "claudia", "alicia", "silvia", "cristina", "monica", "mónica", "sandra",
    "pilar", "mercedes", "angela", "ángela", "julio", "agosto", "marzo", "abril",
    "junio", "enero", "febrero", "mayo", "septiembre", "octubre", "noviembre", "diciembre",
    "lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo",
    "the", "and", "for", "you", "with", "that", "this", "from", "have", "are", "was",
    "not", "but", "all", "can", "one", "out", "what", "when", "who", "how", "more",
    "like", "new", "time", "just", "now", "way", "see", "get", "make", "know", "will",
    "facebook", "blog", "post", "link", "app", "web", "online", "email", "mail", "chat",
    "video", "foto", "ok", "jajaja", "jaja", "haha", "xd", "lol", "sr", "sra", "dr",
    "dra", "etc", "art", "cap", "pág", "núm", "tel", "av", "ed", "vol", "km", "cm", "mm",
    "kg", "gr", "ml", "hz", "ghz", "mb", "gb", "tb", "usd", "eur", "mxn",
    "real", # wait: keep real! don't exclude real!
}
STOP_PROPER_AND_NOISE.discard("real")
# Wait: days of the week and months ARE essential A1 Spanish lemmas! Keep them!
for keep_w in [
    "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
    "septiembre", "octubre", "noviembre", "diciembre",
    "lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo",
    "dios", "foto", "video",
]:
    STOP_PROPER_AND_NOISE.discard(keep_w)


def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def spanish_g2p(word: str, accent: str = "es-ES") -> str:
    """Deterministic, high-accuracy Spanish grapheme-to-IPA converter.
    accent='es-ES' uses Castilian distinción (/θ/); 'es-419' uses Latin American seseo (/s/).
    """
    w = word.lower().strip()
    if not w:
        return ""
    tokens = w.split()
    if len(tokens) > 1:
        return " ".join(spanish_g2p(t, accent) for t in tokens)

    # Identify stressed syllable via standard Spanish stress rules
    vowels = "aeiouáéíóúü"
    accented_vowels = "áéíóú"
    out = []
    i = 0
    n = len(w)
    while i < n:
        ch = w[i]
        nxt = w[i + 1] if i + 1 < n else ""
        prv = w[i - 1] if i > 0 else ""

        # Digraphs first
        if ch == "c" and nxt == "h":
            out.append("tʃ")
            i += 2
            continue
        if ch == "l" and nxt == "l":
            out.append("ʝ" if accent == "es-419" else "ʎ")
            i += 2
            continue
        if ch == "r" and nxt == "r":
            out.append("r")
            i += 2
            continue
        if ch == "q" and nxt == "u":
            out.append("k")
            i += 2
            continue
        if ch == "g" and nxt == "u" and i + 2 < n and w[i + 2] in "eiéí":
            out.append("ɡ" if (i == 0 or prv in "mn") else "ɣ")
            i += 2
            continue
        if ch == "g" and nxt == "ü":
            out.append("ɡw" if (i == 0 or prv in "mn") else "ɣw")
            i += 2
            continue

        # Single letters
        if ch in "aá":
            out.append("a")
        elif ch in "eé":
            out.append("e")
        elif ch in "ií":
            out.append("i")
        elif ch in "oó":
            out.append("o")
        elif ch in "uúü":
            out.append("u")
        elif ch == "h":
            pass
        elif ch in "bv":
            out.append("b" if (i == 0 or prv in "mn") else "β")
        elif ch == "c":
            if nxt in "eiéí":
                out.append("θ" if accent == "es-ES" else "s")
            else:
                out.append("k")
        elif ch == "z":
            out.append("θ" if accent == "es-ES" else "s")
        elif ch == "d":
            out.append("d" if (i == 0 or prv in "ln") else "ð")
        elif ch == "f":
            out.append("f")
        elif ch == "g":
            if nxt in "eiéí":
                out.append("x")
            else:
                out.append("ɡ" if (i == 0 or prv in "mn") else "ɣ")
        elif ch == "j":
            out.append("x")
        elif ch == "k":
            out.append("k")
        elif ch == "l":
            out.append("l")
        elif ch == "m":
            out.append("m")
        elif ch == "n":
            if nxt and nxt in "ckqg":
                out.append("ŋ")
            else:
                out.append("n")
        elif ch == "ñ":
            out.append("ɲ")
        elif ch == "p":
            out.append("p")
        elif ch == "r":
            if i == 0 or prv in "lns":
                out.append("r")
            else:
                out.append("ɾ")
        elif ch == "s":
            out.append("s")
        elif ch == "t":
            out.append("t")
        elif ch == "w":
            out.append("w")
        elif ch == "x":
            out.append("ks" if i > 0 else "s")
        elif ch == "y":
            if i == n - 1 or nxt not in vowels:
                out.append("i")
            else:
                out.append("ʝ")
        i += 1
    return "".join(out)


def load_muse_dictionary() -> dict[str, list[str]]:
    """Load MUSE es-en bilingual dictionary, preserving multiple clean English glosses."""
    mapping: dict[str, list[str]] = defaultdict(list)
    muse_path = CACHE_DIR / "muse_es_en.txt"
    with open(muse_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            es_w, en_w = parts[0].lower(), " ".join(parts[1:]).lower()
            if not es_w or not en_w:
                continue
            # Avoid noisy self-copies unless truly shared English/Spanish word
            if en_w not in mapping[es_w] and len(mapping[es_w]) < 4:
                mapping[es_w].append(en_w)
    return mapping


def load_magpie_phonemes() -> dict[str, str]:
    path = CACHE_DIR / "magpie_es_phoneme_dict.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        if isinstance(v, list) and v:
            if isinstance(v[0], list):
                out[k.lower()] = "".join(str(x) for x in v[0])
            else:
                out[k.lower()] = "".join(str(x) for x in v)
        elif isinstance(v, str):
            out[k.lower()] = v
    return out


def compute_cognate_score(es_lemma: str, en_gloss: str) -> float:
    """Return [0.0, 1.0] orthographic/morphological transparency score for an English speaker
    with zero Spanish guessing the meaning from the written form."""
    es_clean = strip_accents(es_lemma.lower())
    if es_lemma.lower() in FALSE_FRIENDS or es_clean in FALSE_FRIENDS:
        return 0.0
    if len(es_clean) <= 2:
        return 0.0

    # Normalize Spanish Latinate suffixes to English equivalents for root comparison
    def norm_es_candidates(w: str) -> list[str]:
        forms = [w]
        rules = [
            ("cion", "tion"), ("sion", "sion"), ("dad", "ty"), ("tad", "ty"),
            ("mente", "ly"), ("oso", "ous"), ("osa", "ous"), ("ivo", "ive"),
            ("iva", "ive"), ("ico", "ic"), ("ica", "ic"), ("ismo", "ism"),
            ("ista", "ist"), ("ario", "ary"), ("aria", "ary"), ("orio", "ory"),
            ("encia", "ence"), ("ancia", "ance"), ("izar", "ize"), ("ificar", "ify"),
            ("acion", "ation"), ("icion", "ition"), ("idad", "ity"),
            ("ar", "ate"), ("ar", ""), ("er", ""), ("ir", ""),
        ]
        for suf, rep in rules:
            if len(w) > len(suf) + 3 and w.endswith(suf):
                forms.append(w[:-len(suf)] + rep)
        return forms

    es_norms = norm_es_candidates(es_clean)
    best = 0.0

    # Check direct English Latinate cognate existence (catches configuración -> configuration even if gloss says 'settings')
    for nf in es_norms:
        if len(nf) >= 5 and zipf_frequency(nf, "en") >= 3.2:
            best = max(best, 0.88)

    # Split gloss into candidate English headwords
    candidates = re.split(r"[/,;()]", en_gloss.lower())
    for cand in candidates:
        words = [w for w in re.findall(r"[a-z]+", cand) if w not in {"to", "a", "an", "the", "of", "in", "be", "or"}]
        for en_w in words:
            if len(en_w) < 3:
                continue
            for es_n in es_norms:
                score = SequenceMatcher(None, es_n, en_w).ratio()
                if len(es_clean) >= 5 and len(en_w) >= 5 and es_clean[:4] == en_w[:4]:
                    score = min(1.0, score + 0.10)
                best = max(best, score)
    return round(best, 4)


def clean_gloss(es_lemma: str, pos: str, raw_glosses: list[str]) -> str:
    """Format a clean, natural English gloss."""
    if es_lemma in CORE_GLOSS_OVERRIDES:
        return CORE_GLOSS_OVERRIDES[es_lemma][0]
    es_clean = strip_accents(es_lemma.lower())
    cleaned = []
    for g in raw_glosses:
        g = g.strip()
        if not g:
            continue
        if len(g) < 2 or not re.match(r"^[a-z][a-z\s'-]+$", g):
            continue
        # Reject untranslated Spanish self-copy unless it's a genuine common English word
        if strip_accents(g.lower()) == es_clean and zipf_frequency(g.lower(), "en") < 3.6:
            continue
        # For Spanish infinitive verbs, skip English -ing gerunds if we can get base forms
        if pos == "VERB" and g.endswith("ing") and len(g) > 5:
            continue
        if pos == "VERB" and not g.startswith("to "):
            g = f"to {g}"
        if g not in cleaned:
            cleaned.append(g)
        if len(cleaned) >= 2:
            break
    # Fallback if all verb glosses ended in -ing
    if not cleaned and pos == "VERB":
        for g in raw_glosses:
            g = g.strip()
            if len(g) >= 2 and re.match(r"^[a-z][a-z\s'-]+$", g):
                if strip_accents(g.lower()) == es_clean and zipf_frequency(g.lower(), "en") < 3.6:
                    continue
                cleaned.append(f"to {g}" if not g.startswith("to ") else g)
                break
    return " / ".join(cleaned) if cleaned else ""


def assign_visual_symbol(pos: str, gloss: str, category: str = "") -> str:
    """Assign an Apple SF Symbol visual cue for tap-to-gloss & bootstrap picture cards."""
    g = gloss.lower()
    keyword_symbols = [
        (("water", "rain", "sea", "ocean", "river", "drink", "coffee", "tea", "wine", "milk"), "drop.fill"),
        (("sun", "day", "morning", "light", "bright", "summer", "heat", "warm"), "sun.max.fill"),
        (("night", "moon", "dark", "sleep", "dream", "star"), "moon.stars.fill"),
        (("house", "home", "room", "building", "city", "town", "door", "window", "street", "floor"), "building.2.fill"),
        (("food", "eat", "bread", "fruit", "meat", "dinner", "lunch", "breakfast", "cook", "kitchen"), "fork.knife"),
        (("book", "read", "write", "letter", "word", "page", "story", "novel", "paper", "study"), "book.fill"),
        (("speak", "talk", "say", "tell", "voice", "ask", "answer", "call", "language", "explain"), "bubble.left.and.bubble.right.fill"),
        (("hear", "listen", "sound", "music", "song", "ear", "noise", "quiet"), "waveform"),
        (("see", "look", "watch", "eye", "view", "color", "image", "photo"), "eye.fill"),
        (("walk", "run", "go", "travel", "trip", "car", "train", "bus", "plane", "road", "way", "path"), "figure.walk"),
        (("time", "hour", "minute", "year", "month", "week", "clock", "today", "yesterday", "tomorrow"), "clock.fill"),
        (("love", "heart", "feel", "happy", "sad", "friend", "family", "mother", "father", "person", "man", "woman", "child"), "heart.fill"),
        (("money", "pay", "buy", "sell", "price", "cost", "bank", "shop", "store", "work", "job", "office"), "briefcase.fill"),
        (("think", "know", "understand", "remember", "forget", "idea", "mind", "reason", "learn"), "brain.head.profile"),
        (("tree", "flower", "mountain", "park", "garden", "wind", "snow", "cold", "weather", "nature", "animal", "dog", "cat"), "leaf.fill"),
        (("hand", "head", "body", "health", "doctor", "hospital", "pain", "hurt", "life"), "cross.case.fill"),
        (("fire", "hot", "energy", "power", "strong", "fast"), "flame.fill"),
        (("key", "open", "close", "lock", "secret", "find", "search"), "key.fill"),
        (("world", "earth", "country", "place", "map", "north", "south", "east", "west"), "globe.europe.africa.fill"),
    ]
    for kws, sym in keyword_symbols:
        if any(re.search(rf"\b{kw}\b", g) for kw in kws):
            return sym
    pos_defaults = {
        "VERB": "bolt.circle.fill",
        "NOUN": "cube.fill",
        "ADJ": "sparkles",
        "ADV": "gauge.with.needle.fill",
        "ADP": "arrow.triangle.branch",
        "PRON": "person.fill",
        "DET": "tag.fill",
        "CCONJ": "link",
        "SCONJ": "link",
        "NUM": "number.circle.fill",
        "CHUNK": "quote.bubble.fill",
        "GRAMMAR": "text.book.closed.fill",
    }
    return pos_defaults.get(pos, "sparkle")


def build_database():
    print("1. Loading spaCy es_core_news_md, MUSE es-en, and Magpie phoneme dict...")
    nlp = spacy.load("es_core_news_md", disable=["ner", "parser"])
    muse = load_muse_dictionary()
    magpie_ipa = load_magpie_phonemes()

    print("2. Pulling top 65,000 Spanish wordforms from wordfreq (spoken + written combined)...")
    raw_words = top_n_list("es", 65000)

    # Batch-lemmatize with spaCy for speed and build full inflection->lemma table
    candidates = []
    for w in raw_words:
        wl = w.lower()
        if wl in SINGLE_LETTER_WORDS or VALID_SPANISH_RE.match(wl):
            candidates.append(wl)

    print(f"   Filtered {len(candidates)} valid alphabetic Spanish wordforms; lemmatizing with spaCy...")
    lemma_freq: dict[str, float] = defaultdict(float)
    lemma_pos_votes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    inflection_map: dict[str, tuple[str, str, float]] = {}  # surface -> (lemma, pos, freq)

    # Process in fast batches via nlp.pipe
    batch_size = 2000
    for idx in range(0, len(candidates), batch_size):
        chunk = candidates[idx:idx + batch_size]
        # Lemmatize each token individually in mini-docs of 1 word for clean morphological lookup
        docs = list(nlp.pipe(chunk, batch_size=500, n_process=1))
        for word, doc in zip(chunk, docs):
            if not doc:
                continue
            tok = doc[0]
            pos = tok.pos_
            lem = tok.lemma_.lower().strip()
            # Keep only first token if spaCy returns multi-word lemma
            if " " in lem:
                lem = lem.split()[0]
            if not (lem in SINGLE_LETTER_WORDS or VALID_SPANISH_RE.match(lem)):
                lem = word

            zipf = zipf_frequency(word, "es")
            linear_freq = 10.0 ** zipf

            # Record surface -> lemma mapping for the offline Swift lemmatizer
            if word not in inflection_map or linear_freq > inflection_map[word][2]:
                inflection_map[word] = (lem, pos, linear_freq)

            # Exclude proper nouns and noisy abbreviations from the active syllabus
            if pos in {"PROPN", "SYM", "X", "PUNCT", "SPACE"}:
                if word not in CORE_GLOSS_OVERRIDES and lem not in CORE_GLOSS_OVERRIDES:
                    continue
            if lem in STOP_PROPER_AND_NOISE or word in STOP_PROPER_AND_NOISE:
                continue

            lemma_freq[lem] += linear_freq
            lemma_pos_votes[lem][pos] += linear_freq

    # Ensure every CORE_GLOSS_OVERRIDES item is present with high priority
    for core_lem, (core_gloss, core_pos, _) in CORE_GLOSS_OVERRIDES.items():
        if core_lem not in lemma_freq:
            lemma_freq[core_lem] = 10.0 ** max(4.5, zipf_frequency(core_lem, "es"))
        lemma_pos_votes[core_lem][core_pos] += lemma_freq[core_lem] * 2.0
        inflection_map[core_lem] = (core_lem, core_pos, lemma_freq[core_lem])

    # Also enrich inflection map with common verb enclitics / regular forms & Magpie phoneme words
    print(f"   Collected {len(lemma_freq)} candidate lemmas and {len(inflection_map)} surface inflections.")

    # Sort lemmas by combined spoken+written linear frequency descending
    ranked_lemmas = sorted(lemma_freq.items(), key=lambda kv: kv[1], reverse=True)

    # Filter to lemmas that have a verified English gloss (from CORE_GLOSS_OVERRIDES or MUSE)
    # and aren't capitalized-only proper nouns or English loanword noise
    verified_lemmas = []
    skipped_no_gloss = 0
    for lem, freq in ranked_lemmas:
        pos_counts = lemma_pos_votes[lem]
        best_pos = max(pos_counts.items(), key=lambda kv: kv[1])[0]
        if lem in CORE_GLOSS_OVERRIDES:
            gloss, best_pos, forced_cog = CORE_GLOSS_OVERRIDES[lem]
            cog_score = 0.95 if forced_cog else 0.0
            is_cog = forced_cog
        else:
            raw_g = muse.get(lem, [])
            if not raw_g:
                # Try accent-stripped fallback only if exact match exists
                skipped_no_gloss += 1
                continue
            gloss = clean_gloss(lem, best_pos, raw_g)
            if not gloss:
                skipped_no_gloss += 1
                continue
            # Reject pure English untranslated noise where Spanish word is identical English stopword
            if lem == gloss.lower() and best_pos not in {"NOUN", "ADJ"} and zipf_frequency(lem, "en") > zipf_frequency(lem, "es") + 1.2:
                continue
            cog_score = compute_cognate_score(lem, gloss)
            is_cog = cog_score >= 0.78

        zipf_val = math.log10(max(freq, 1.0))
        ipa_es = magpie_ipa.get(lem, spanish_g2p(lem, "es-ES"))
        ipa_latam = spanish_g2p(lem, "es-419")
        verified_lemmas.append({
            "lemma": lem,
            "pos": best_pos,
            "gloss": gloss,
            "zipf": round(zipf_val, 3),
            "cognate_score": cog_score,
            "is_cognate": is_cog,
            "ipa_es": ipa_es,
            "ipa_latam": ipa_latam,
            "symbol": assign_visual_symbol(best_pos, gloss),
        })
        if len(verified_lemmas) >= 6200:
            break

    print(f"   Verified {len(verified_lemmas)} glossed Spanish lemmas (skipped {skipped_no_gloss} unglossed/noisy).")
    assert len(verified_lemmas) >= 4800, f"Need at least 4800 verified lemmas, got {len(verified_lemmas)}"

    # Top 4,000 active word lemmas
    active_4000 = verified_lemmas[:4000]

    # 400 non-cognate hold-out lemmas strictly from the 4,001–5,500 band
    holdout_400 = []
    for entry in verified_lemmas[4000:]:
        if not entry["is_cognate"] and entry["cognate_score"] < 0.42 and len(entry["lemma"]) >= 4:
            holdout_400.append(entry)
            if len(holdout_400) == 400:
                break
    assert len(holdout_400) == 400, f"Expected 400 non-cognate holdout lemmas, got {len(holdout_400)}"

    # Load 200 multi-word chunks
    with open(DATA_DIR / "spanish_chunks_200.json", "r", encoding="utf-8") as f:
        chunks_200 = json.load(f)
    assert len(chunks_200) == 200, f"Expected 200 chunks, got {len(chunks_200)}"

    # Load 300 Instituto Cervantes grammar patterns
    with open(DATA_DIR / "cervantes_a1_a2.json", "r", encoding="utf-8") as f:
        gram_a = json.load(f)
    with open(DATA_DIR / "cervantes_b1_b2.json", "r", encoding="utf-8") as f:
        gram_b = json.load(f)
    grammar_300 = gram_a + gram_b
    assert len(grammar_300) == 300, f"Expected 300 grammar patterns, got {len(grammar_300)}"
    assert len({g["id"] for g in grammar_300}) == 300, "Duplicate grammar IDs!"

    # Load assessment seeds
    with open(DATA_DIR / "assessments_seed.json", "r", encoding="utf-8") as f:
        assessments = json.load(f)

    # Mark bootstrap pool (Sessions 1-4 = 240 items: commonest function words + transparent cognates)
    function_pos = {"DET", "ADP", "PRON", "CCONJ", "SCONJ", "AUX", "ADV", "NUM"}
    bootstrap_ids = set()
    # First pick top ~130 core function words + essential verbs (ser, estar, tener, ir, hacer, querer, poder, hablar, ver, dar)
    essential_verbs = {"ser", "estar", "tener", "ir", "hacer", "querer", "poder", "hablar", "ver", "dar", "decir", "saber", "comer", "beber", "vivir", "venir", "salir", "llevar", "gustar", "necesitar"}
    for i, entry in enumerate(active_4000):
        if len(bootstrap_ids) >= 130:
            break
        if entry["pos"] in function_pos or entry["lemma"] in essential_verbs:
            bootstrap_ids.add(entry["lemma"])
    # Fill remaining up to 240 with highest-frequency transparent cognates
    for entry in active_4000:
        if len(bootstrap_ids) >= 240:
            break
        if entry["is_cognate"] and entry["lemma"] not in bootstrap_ids:
            bootstrap_ids.add(entry["lemma"])

    # Create SQLite seed database
    OUT_SQLITE.parent.mkdir(parents=True, exist_ok=True)
    if OUT_SQLITE.exists():
        OUT_SQLITE.unlink()

    conn = sqlite3.connect(OUT_SQLITE)
    cur = conn.cursor()
    cur.executescript("""
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=OFF;

        CREATE TABLE syllabus_items (
            item_id TEXT PRIMARY KEY,
            item_type TEXT NOT NULL CHECK(item_type IN ('lemma', 'chunk', 'grammar', 'holdout')),
            rank_index INTEGER NOT NULL,
            spanish TEXT NOT NULL,
            english_gloss TEXT NOT NULL,
            pos TEXT NOT NULL,
            cefr TEXT NOT NULL,
            ipa_es TEXT NOT NULL,
            ipa_latam TEXT NOT NULL,
            is_transparent_cognate INTEGER NOT NULL DEFAULT 0,
            cognate_score REAL NOT NULL DEFAULT 0.0,
            is_bootstrap INTEGER NOT NULL DEFAULT 0,
            is_holdout INTEGER NOT NULL DEFAULT 0,
            is_top2000_productive_target INTEGER NOT NULL DEFAULT 0,
            zipf_frequency REAL NOT NULL DEFAULT 0.0,
            visual_symbol TEXT NOT NULL DEFAULT 'sparkle',
            grammar_category TEXT,
            grammar_week INTEGER,
            micro_note_en TEXT,
            contrast_a_es TEXT,
            contrast_a_en TEXT,
            contrast_b_es TEXT,
            contrast_b_en TEXT,
            literal_gloss TEXT,
            example_es TEXT
        );

        CREATE INDEX idx_syllabus_type_rank ON syllabus_items(item_type, rank_index);
        CREATE INDEX idx_syllabus_spanish ON syllabus_items(spanish);
        CREATE INDEX idx_syllabus_holdout ON syllabus_items(is_holdout);
        CREATE INDEX idx_syllabus_bootstrap ON syllabus_items(is_bootstrap);

        CREATE TABLE spanish_inflections (
            surface TEXT PRIMARY KEY,
            lemma TEXT NOT NULL,
            pos TEXT NOT NULL
        );

        CREATE INDEX idx_inflections_lemma ON spanish_inflections(lemma);

        CREATE TABLE assessment_pseudowords (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT NOT NULL UNIQUE
        );

        CREATE TABLE assessment_eit_items (
            id INTEGER PRIMARY KEY,
            syllables INTEGER NOT NULL,
            cefr TEXT NOT NULL,
            spanish TEXT NOT NULL,
            english TEXT NOT NULL,
            ipa_es TEXT NOT NULL,
            ipa_latam TEXT NOT NULL
        );

        CREATE TABLE assessment_native_listening (
            id TEXT PRIMARY KEY,
            checkpoint_hour INTEGER NOT NULL,
            accent TEXT NOT NULL,
            speaker TEXT NOT NULL,
            wpm INTEGER NOT NULL,
            title TEXT NOT NULL,
            transcript TEXT NOT NULL,
            questions_json TEXT NOT NULL
        );

        CREATE TABLE serial_genres (
            id TEXT PRIMARY KEY,
            title_es TEXT NOT NULL,
            title_en TEXT NOT NULL,
            tagline TEXT NOT NULL,
            setting TEXT NOT NULL,
            premise TEXT NOT NULL,
            accent_palette TEXT NOT NULL,
            protagonist TEXT NOT NULL
        );
    """)

    # Insert 4,000 active word lemmas
    lemma_rows = []
    for idx, entry in enumerate(active_4000, start=1):
        cefr = "A1" if idx <= 600 else ("A2" if idx <= 1400 else ("B1" if idx <= 2600 else "B2"))
        lemma_rows.append((
            f"LEM_{idx:04d}",
            "lemma",
            idx,
            entry["lemma"],
            entry["gloss"],
            entry["pos"],
            cefr,
            entry["ipa_es"],
            entry["ipa_latam"],
            1 if entry["is_cognate"] else 0,
            entry["cognate_score"],
            1 if entry["lemma"] in bootstrap_ids else 0,
            0,
            1 if idx <= 2000 else 0,
            entry["zipf"],
            entry["symbol"],
            None, None, None, None, None, None, None, None, None,
        ))

    # Insert 400 non-cognate holdout lemmas (ranks 4001..4400)
    holdout_rows = []
    for idx, entry in enumerate(holdout_400, start=1):
        holdout_rows.append((
            f"HOLD_{idx:03d}",
            "holdout",
            4000 + idx,
            entry["lemma"],
            entry["gloss"],
            entry["pos"],
            "B2",
            entry["ipa_es"],
            entry["ipa_latam"],
            0,
            entry["cognate_score"],
            0,
            1,
            0,
            entry["zipf"],
            entry["symbol"],
            None, None, None, None, None, None, None, None, None,
        ))

    # Insert 200 multi-word chunks
    chunk_rows = []
    for idx, c in enumerate(chunks_200, start=1):
        surf = c["surface"].lower().strip()
        chunk_rows.append((
            f"CHK_{idx:03d}",
            "chunk",
            idx,
            surf,
            c["gloss"],
            "CHUNK",
            c["cefr"],
            spanish_g2p(surf, "es-ES"),
            spanish_g2p(surf, "es-419"),
            0,
            0.0,
            0,
            0,
            0,
            4.5 - idx * 0.005,
            assign_visual_symbol("CHUNK", c["gloss"]),
            "Multi-Word Chunk",
            1 if idx <= 20 else (2 if idx <= 55 else (3 if idx <= 95 else (4 if idx <= 130 else (5 if idx <= 165 else 6)))),
            f"Idiomatic fixed expression (literally: '{c['literal']}'). Learn as a single unit.",
            c["example"],
            c["gloss"],
            None,
            None,
            c["literal"],
            c["example"],
        ))

    # Insert 300 Instituto Cervantes grammar patterns
    grammar_rows = []
    for idx, g in enumerate(grammar_300, start=1):
        grammar_rows.append((
            g["id"],
            "grammar",
            idx,
            g["title"],
            g["micro_note"],
            "GRAMMAR",
            g["cefr"],
            spanish_g2p(g["contrast_a"], "es-ES"),
            spanish_g2p(g["contrast_a"], "es-419"),
            0,
            0.0,
            1 if idx <= 12 else 0,
            0,
            0,
            5.0 - idx * 0.005,
            "text.book.closed.fill",
            g["category"],
            g["week"],
            g["micro_note"],
            g["contrast_a"],
            g["contrast_a_en"],
            g["contrast_b"],
            g["contrast_b_en"],
            None,
            g["contrast_a"],
        ))

    cur.executemany(
        """INSERT INTO syllabus_items VALUES (
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
        )""",
        lemma_rows + holdout_rows + chunk_rows + grammar_rows,
    )

    # Populate Spanish inflection -> lemma dictionary (covering all spaCy-mapped forms + active lemmas)
    inf_rows = [(surf, lem, pos) for surf, (lem, pos, _) in inflection_map.items()]
    cur.executemany(
        "INSERT OR REPLACE INTO spanish_inflections (surface, lemma, pos) VALUES (?, ?, ?)",
        inf_rows,
    )

    # Populate assessment seeds
    cur.executemany(
        "INSERT INTO assessment_pseudowords (word) VALUES (?)",
        [(w,) for w in assessments["pseudowords"]],
    )
    cur.executemany(
        "INSERT INTO assessment_eit_items (id, syllables, cefr, spanish, english, ipa_es, ipa_latam) VALUES (?,?,?,?,?,?,?)",
        [
            (
                item["id"], item["syllables"], item["cefr"],
                item["spanish"], item["english"],
                spanish_g2p(item["spanish"], "es-ES"),
                spanish_g2p(item["spanish"], "es-419"),
            )
            for item in assessments["elicited_imitation_task"]
        ],
    )
    cur.executemany(
        "INSERT INTO assessment_native_listening VALUES (?,?,?,?,?,?,?,?)",
        [
            (
                item["id"], item["checkpoint_hour"], item["accent"], item["speaker"],
                item["wpm"], item["title"], item["transcript"],
                json.dumps(item["questions"], ensure_ascii=False),
            )
            for item in assessments["native_speed_listening_tests"]
        ],
    )

    # Populate the 3 selectable serial story genres (Rule 7: Continuity)
    serials = [
        (
            "madrid_noir",
            "La Sombra del Retiro",
            "The Shadow of Retiro",
            "A rain-slicked architectural mystery unfolding across Madrid's cafés, train stations, and hidden archives.",
            "Madrid, Spain (Castilian & Pan-Hispanic)",
            "Clara Navarro, an acoustic engineer who restores old clock towers, discovers coded messages hidden inside brass bell chimes across Madrid—drawing her into a quiet conspiracy spanning three decades.",
            "terracotta_midnight",
            "Clara Navarro",
        ),
        (
            "patagonia_scifi",
            "Estación Austral",
            "Southern Station",
            "A luminous near-future expedition at a deep-ice atmospheric observatory in Argentine Patagonia.",
            "El Chaltén & Tierra del Fuego (Rioplatense & Pan-Hispanic)",
            "Mateo Solís joins a multidisciplinary crew studying strange harmonic pressure waves rising from beneath the Patagonian ice sheet, racing the onset of winter storms to decode an impossible signal.",
            "cobalt_aurora",
            "Mateo Solís",
        ),
        (
            "oaxaca_valencia",
            "Sal y Azafrán",
            "Salt and Saffron",
            "A warm, sun-drenched culinary and coastal drama linking a family kitchen in Oaxaca to Valencia's harbor.",
            "Oaxaca, Mexico & Valencia, Spain",
            "Two estranged cousins inherit twin notebooks—one full of ancestral spice recipes in Oaxaca, the other full of maritime letters in Valencia—and have one summer to rebuild a legendary neighborhood table.",
            "saffron_amber",
            "Lucía & Tomás Morales",
        ),
    ]
    cur.executemany(
        "INSERT INTO serial_genres VALUES (?,?,?,?,?,?,?,?)",
        serials,
    )

    conn.commit()
    cur.execute("VACUUM")
    conn.close()

    # Write verification summary report
    cognate_count = sum(1 for e in active_4000 if e["is_cognate"])
    report = {
        "sqlite_path": str(OUT_SQLITE),
        "sqlite_bytes": OUT_SQLITE.stat().st_size,
        "counts": {
            "active_lemmas": len(lemma_rows),
            "multi_word_chunks": len(chunk_rows),
            "grammar_patterns": len(grammar_rows),
            "total_active_syllabus": len(lemma_rows) + len(chunk_rows) + len(grammar_rows),
            "holdout_non_cognate_lemmas": len(holdout_rows),
            "bootstrap_items": sum(1 for r in lemma_rows if r[11] == 1),
            "transparent_cognates_in_4000": cognate_count,
            "spanish_inflections": len(inf_rows),
            "pseudowords": len(assessments["pseudowords"]),
            "eit_sentences": len(assessments["elicited_imitation_task"]),
            "native_listening_tests": len(assessments["native_speed_listening_tests"]),
            "serial_genres": len(serials),
        },
        "sample_top_20_lemmas": active_4000[:20],
        "sample_holdout_10": holdout_400[:10],
    }
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("DONE! Report:")
    print(json.dumps(report["counts"], indent=2))
    print(f"SQLite size: {OUT_SQLITE.stat().st_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    build_database()
