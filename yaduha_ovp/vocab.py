from yaduha.language import VocabEntry

NOUNS = [
    VocabEntry(english="coyote", target="isha'"),
    VocabEntry(english="vulture", target="wiho"),
    VocabEntry(english="dog", target="ishapugu"),
    VocabEntry(english="cat", target="kidi'"),
    VocabEntry(english="horse", target="pugu"),
    VocabEntry(english="rice", target="wai"),
    VocabEntry(english="pinenuts", target="tüba"),
    VocabEntry(english="corn", target="maishibü"),
    VocabEntry(english="water", target="paya"),
    VocabEntry(english="river", target="payahuupü"),
    VocabEntry(english="chair", target="katünu"),
    VocabEntry(english="mountain", target="toyabi"),
    VocabEntry(english="food", target="tuunapi"),
    VocabEntry(english="tree", target="pasohobü"),
    VocabEntry(english="house", target="nobi"),
    VocabEntry(english="wickiup", target="toni"),
    VocabEntry(english="cup", target="apo"),
    VocabEntry(english="wood", target="küna"),
    VocabEntry(english="rock", target="tübbi"),
    VocabEntry(english="cottontail", target="tabuutsi'"),
    VocabEntry(english="jackrabbit", target="kamü"),
    VocabEntry(english="apple", target="aaponu'"),
    VocabEntry(english="weasle", target="tüsüga"),
    VocabEntry(english="lizard", target="mukita"),
    VocabEntry(english="mosquito", target="wo'ada"),
    VocabEntry(english="bird_snake", target="wükada"),
    VocabEntry(english="worm", target="wo'abi"),
    VocabEntry(english="squirrel", target="aingwü"),
    VocabEntry(english="bird", target="tsiipa"),
    VocabEntry(english="earth", target="tüwoobü"),
    VocabEntry(english="coffee", target="koopi'"),
    VocabEntry(english="bear", target="pahabichi"),
    VocabEntry(english="fish", target="pagwi"),
    VocabEntry(english="tail", target="kwadzi"),
    VocabEntry(english="raccoon", target="padaka'i"),
    VocabEntry(english="chipmunk", target="taba'ya"),
    VocabEntry(english="knife", target="wihi"),
]

TRANSITIVE_VERBS = [
    VocabEntry(english="eat", target="tüka"),
    VocabEntry(english="see", target="puni"),
    VocabEntry(english="drink", target="hibi"),
    VocabEntry(english="hear", target="naka"),
    VocabEntry(english="smell", target="kwana"),
    VocabEntry(english="hit", target="kwati"),
    VocabEntry(english="talk_to", target="yadohi"),
    VocabEntry(english="chase", target="naki"),
    VocabEntry(english="climb", target="tsibui"),
    VocabEntry(english="cook", target="sawa"),
    VocabEntry(english="read", target="nia"),
    VocabEntry(english="write", target="mui"),
    VocabEntry(english="visit", target="nobini"),
    VocabEntry(english="find", target="tama'i"),
]

INTRANSITIVE_VERBS = [
    VocabEntry(english="sit", target="katü"),
    VocabEntry(english="sleep", target="üwi"),
    VocabEntry(english="sneeze", target="kwisha'i"),
    VocabEntry(english="run", target="poyoha"),
    VocabEntry(english="go", target="mia"),
    VocabEntry(english="walk", target="hukaw̃ia"),
    VocabEntry(english="stand", target="wünü"),
    VocabEntry(english="lie_down", target="habi"),
    VocabEntry(english="talk", target="yadoha"),
    VocabEntry(english="fall", target="kwatsa'i"),
    VocabEntry(english="work", target="waakü"),
    VocabEntry(english="smile", target="wükihaa"),
    VocabEntry(english="sing", target="hubiadu"),
    VocabEntry(english="laugh", target="nishua'i"),
    VocabEntry(english="climb", target="tsibui"),
    VocabEntry(english="play", target="tübinohi"),
    VocabEntry(english="fly", target="yotsi"),
    VocabEntry(english="jump", target="yotsi"),
    VocabEntry(english="dance", target="nüga"),
    VocabEntry(english="swim", target="pahabi"),
    VocabEntry(english="read", target="tünia"),
    VocabEntry(english="write", target="tümui"),
    VocabEntry(english="chirp", target="tsiipe'i"),
]

# Kinship terms data - just the raw vocabulary
KINSHIP_TERMS_DATA = [
    {
        "english": "mother",
        "unpossessed": "piabi",
        "possessed_stem": "bia",
    },
    # Add more as you learn them:
    # {"english": "father", "unpossessed": "...", "possessed_stem": "..."},
    # {"english": "grandmother", "unpossessed": "...", "possessed_stem": "..."},
]

# Possessive prefixes for kinship terms
POSSESSIVE_PREFIXES = {
    "1sg": "i-",   # my
    "2sg": "ü-",   # your
    "3sg": "ma-",  # his/her
    # Add more as needed
}
