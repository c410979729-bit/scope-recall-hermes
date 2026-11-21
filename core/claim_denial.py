"""Match explicit denials to a held value without changing the claim or its evidence."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .fact_evidence import stem
from .source_qualification import AUTHORITY_QUESTION

#: The verbs of being, having, liking and doing a Chinese negation opens a value with ("不是 X", "没有车", "不吃辣",
#: "不喜欢用 X", "不在线"); before any other word a negator is part of that word (不错, 不过, 没问题, 不客气, 不同意,
#: 不知道, 不用谢, 不在乎).
_DENIED_VERB = (
    r"是|有(?!问题|关系|事)|喜欢|爱|想(?!到)|要(?!紧)|需要|用(?!谢)|使用|吃|喝|抽|吸|穿|戴|碰|玩|开|住|睡|养|信|怕"
    r"|在(?!乎|意)|去|做|坐|骑|买|选|看|听"
)
#: A negation at the opening of a value, after the adverbs that may lead it ("一点也不喜欢", "really don't"): a
#: Chinese negator with the degree or the time it carries (不是很, 不太, 不再, 别再) before a verb of being, having,
#: liking or doing; an outright rejection ("并非 X", "讨厌 X", "受不了 X", "can't stand X", "dislikes X"); or an
#: English negator ("not X", "isn't X", "don't have a car", "never drink coffee", "no longer uses X").  不一定 doubts
#: rather than negates, and an English negator before a word of knowing, minding, applying or contrast negates nothing
#: ("not bad", "not only X but Y", "don't know", "not sure", "not applicable", "doesn't apply", "isn't it X").  Joined
#: to the word after it, a negator is part of that word, as in NOT_STARTED, no-op and never-ending.
_NEGATOR = re.compile(
    r"^\s*(?:(?:一点[也都]|根本|完全|从来|从|向来|也|都|还|并|已经|再也|很|真|实在|一直|平时|现在)\s*)*"
    r"(?:(?P<outright>并非|绝非|讨厌|厌恶|反感|受不了)(?=\s*[^\s。，！？,.!?])"
    r"|(?P<negator>不|没|别|未曾|尚未|未)(?:是(?=很|太|那么|特别))?(?P<modal>(?:太|大|很|那么|特别|怎么|常|再|会|能)*)"
    rf"(?=(?P<verb>{_DENIED_VERB})\s*[^\s。，！？,.!?]))"
    r"|^\s*(?:(?:really|just|still|simply|definitely|absolutely|is|are|was|were|do|does|did)\s+)*"
    r"(?:(?:not|never|no\s+longer|cannot|can['’]?t|\w+n['’]t)\s+"
    r"(?!(?:bad|only|just|merely|sure|certain|know|knows|knew|think|mind|care|matter|remember|applicable|apply"
    r"|applies|it|that|this|there|a\s+problem)\b)"
    r"|(?P<dislike>dislikes?|hates?)\s+)(?=[^\s.,!?])",
    re.I,
)
#: The Chinese verbs a 不 or a 别 before them negates as a state of the subject rather than a choice: being somewhere
#: (不在线, 不在家), having, and opening or enabling (不开放).
_ZH_STATE_VERBS = frozenset({"在", "有", "开"})
#: The degree words a 不是 may carry: before a word of state they make a degree of it (不是很稳定), and only a verb of
#: liking after them a rejection (不是很喜欢 X).
_ZH_DEGREE = re.compile(r"很|太|那么|特别")
#: English words that may stand between a negator and the word it negates and name nothing themselves ("not
#: currently", "isn't being", "not a").  With nothing after them, the negation is a polar answer ("not really").
_EN_UNSTRESSED = frozenset(
    "a an the be been being currently fully always very too so quite really entirely completely totally".split()
)
#: English words of state that no ending marks: adjectives and polar answers (not null, not ready, not online, not
#: empty, not enough) and irregular past participles (not found, not set, not done, not built, not run).
_EN_STATE_WORDS = frozenset(
    """null nil none ready online offline empty free open live valid invalid up down due busy idle full safe secure
    true false ok okay fine good bad great well enough much many here away healthy dirty flaky noisy steady easy early
    fast slow main
    likely read-only up-to-date done set found known shown seen given taken broken chosen written built run made sent
    paid held kept left lost met read said sold told won begun gone got gotten hidden frozen driven eaten proven spoken
    stolen worn torn grown thrown drawn""".split()
)
#: Endings an English adjective or participle carries strongly enough to describe the subject (deployed, running,
#: available, active, continuous, useful, stateless, optional, public, persistent, temporary); -ment ends a noun
#: (deployment).  ible and ant also end names, so they leave the reading open (``_EN_OPEN_ENDING``).
_EN_STATE_ENDING = re.compile(r"(?:ed|ing|able|ive|ous|ful|less|al|ic|(?<!m)ent|ary)$")
#: What case cannot settle: history may identify it, and a capital may not name it.  ible and ant end a
#: description or a name (compatible, Ansible, vagrant).  default is a status a slot may hold: listed with the state
#: words, "not default" would stay a value after a held default, and a capital would deny "not Default" with none.
_EN_OPEN_WORDS = frozenset({"default"})
_EN_OPEN_ENDING = re.compile(r"(?:ible|ant)$")
#: English verbs of having, holding and doing what a thing is for: negated, they say whether the subject has or does
#: it, a state ("doesn't have a GPU", "doesn't support IPv6", "doesn't work", "won't start").
_EN_HAVING = frozenset(
    "have has had own support include contain allow accept work run exist respond load start connect build compile pass"
    " fit scale".split()
)
#: English verbs of liking, wanting, choosing, using and taking in: negated, they reject what they name ("don't like
#: PowerShell", "no longer uses Jenkins", "never drink coffee", "can't stand X").
_EN_REJECTING = frozenset(
    "like love enjoy prefer want need use choose pick eat drink smoke wear play watch listen drive ride buy take go"
    " stand bear".split()
)
#: How an English negator reads the verb after it: one a subject cannot do (can't connect), one that did not happen
#: (didn't pass), and one a subject does not, will not or no longer does (doesn't support, won't start, never drink).
_EN_UNABLE = re.compile(r"\bcan(?:not|['’]t|\s+not)\b|\bcould(?:n['’]t|\s+not)\b", re.I)
_EN_DID_NOT = re.compile(r"\bdid(?:n['’]t|\s+not)\b", re.I)
_EN_VERBAL = re.compile(
    r"\b(?:do|does|will|wo|would|should|has|have|had)(?:n['’]t|\s+not)\b|\bnever\b|\bno\s+longer\b", re.I
)
#: A complete number-word quantity names a scalar alternative, as a digit does; an age or degree description
#: ("not one week old", 不是一周的) does not match. Missing quantity forms remain values.
_NAMED_QUANTITY = re.compile(
    r"(?:(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen"
    r"|sixteen|seventeen|eighteen|nineteen|(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?)\s+"
    r"(?:milliseconds?|seconds?|minutes?|hours?|days?|weeks?|months?|years?|bytes?|kilobytes?|megabytes?|gigabytes?"
    r"|terabytes?|requests?)|[零〇一二两三四五六七八九十百千万亿]+(?:毫秒|秒|分钟|小时|天|周|个月|月|年|字节|次))",
    re.I,
)
#: Where a denial goes on to say what holds instead: "not X but Y", 不是 X 而是 Y, 不是 X，是 Y, 不再用 X 改用 Y.
_CONTRAST = re.compile(
    r"\s*(?:[，,;；]\s*(?:而是|是|改用|改为|改成|换成|换用|换为|用)|而是|改用|改为|改成|换成|换用|换为"
    r"|,?\s*\b(?:but|instead)\b)",
    re.I,
)
#: A particle or a word of time a denial may end its target with, which neither names nor narrows what it denies (了,
#: "any more", "again").
_SINCE = re.compile(
    r"(?:\s*(?:了|啦|呀|啊|以后|now|already|anymore|any\s+(?:more|longer)|no\s+more|at\s+all|(?:ever\s+)?again"
    r"|from\s+now\s+on))+[\s。.!！]*$",
    re.I,
)
#: Kinds whose slot holds a value, so a denial can only say what it is not.  A constraint's negation is the rule
#: itself ("never push to main"), and a decision not to do something is the decision.
_VALUE_SLOT_KINDS = frozenset({"fact", "preference"})
#: Why a denial's own version is a proposal: it denies another value than the one its slot holds; or it is no value
#: of its slot (``mutate.denial_verdict``).
DENIAL_REASON = "denial_of_another_value"
NOT_A_VALUE_REASON = "denial_not_a_value"
#: Why a value ended: a source denies it.
DENIED_REASON = "value_denied"
#: A word of a value as values are compared: one CJK character, or an identifier whole with the marks it carries
#: (v3.1, my_var, ops-pager, a/b, #ops-pager, @owner).
_VALUE_WORD = re.compile(r"[\u3400-\u9fff]|[#@]?[^\W\u3400-\u9fff](?:[./-]?[^\W\u3400-\u9fff])*")
_ARTICLES = frozenset({"a", "an", "the"})


def _value_words(text: str) -> tuple[str, ...]:
    words = _VALUE_WORD.findall(unicodedata.normalize("NFKC", text).casefold())
    return tuple(stem(word) for word in words if word not in _ARTICLES)


#: The copula a denial puts before what it denies ("不是 X", "is not X"), and the verbs it may name it under ("不喜欢用
#: X", "don't like X", "can't stand X"), as ``_value_words`` gives them.
_COPULA = frozenset(_value_words("是 is are was were am be"))
_DENYING_VERBS = _COPULA | frozenset(
    _value_words(
        "有 喜欢 爱 想 要 需要 用 使用 吃 喝 抽 吸 穿 戴 碰 玩 开 住 睡 养 信 怕 在 去 做 坐 骑 买 选 看 听"
        " have has having like liking love want need use using eat drink smoke wear drive play own live work go take"
        " prefer enjoy stand"
    )
)


@dataclass(frozen=True)
class Denial:
    """A denial read from a claim's value (``read_denial``): the words of the target it denies, as ``_value_words``
    gives them and without the time words after it, and whether it goes on to say what holds instead."""

    target: tuple[str, ...]
    contrast: bool

    @property
    def names_a_value(self) -> bool:
        """Whether the target holds more than the verbs of its construction: 不吃了 and "never again" name nothing."""
        return any(word not in _DENYING_VERBS for word in self.target)


def _set_in_capitals(value: str, negator: str) -> bool:
    """Whether a value opening with an English negator is set in capitals, every letter of it: a term as a keyword, a
    status or a flag is written (NOT NULL, NOT FOUND), not a sentence's denial.  Capitals on the negator alone, before
    words in their own case, stress a denial ("NOT Jenkins any more"); a Chinese negator has no case."""
    cased = [char for char in value if char.lower() != char.upper()]
    return any(char.lower() != char.upper() for char in negator) and all(char.isupper() for char in cased)


def _negation(payload) -> re.Match | None:
    """The negation a fact's or a preference's value opens with, unless the value asks ("isn't it X?")."""
    value = str(payload.get("value_text") or "")
    if payload.get("kind") not in _VALUE_SLOT_KINDS or AUTHORITY_QUESTION.search(value):
        return None
    return _NEGATOR.match(value)


def _lemmas(word: str) -> set[str]:
    """The forms an English verb may have before its ending: likes, liking and liked give like, uses and using use,
    running run."""
    forms = {word}
    for ending, restored in (("ies", "y"), ("es", ""), ("s", ""), ("ing", ""), ("ing", "e"), ("ed", ""), ("ed", "e")):
        if word.endswith(ending) and len(word) > len(ending) + 1:
            base = word[: -len(ending)] + restored
            forms.add(base)
            if not restored and len(base) > 2 and base[-1] == base[-2]:
                forms.add(base[:-1])
    return forms


def _written_as_a_name(word: str) -> bool:
    """Digits, identifier marks and camelCase positively identify a target; capitals alone also spell states."""
    return any(char.isdigit() or char in "#@./_" for char in word) or re.search(r"[a-z][A-Z]", word) is not None


def _english_evidence(negator: str, target: str) -> bool | None:
    """``_target_evidence`` for an English negation, read from the first word it negates that names something.  A
    negation of a state not yet reached (\"not yet\") or of unstressed words alone (\"not really\") describes."""
    words = [word for word in (part.strip(",;:!?\"'()[]").rstrip(".") for part in target.split()) if word]
    if not words or words[0].casefold() == "yet":
        return False
    while words and words[0].casefold() in _EN_UNSTRESSED:
        words = words[1:]
    return _english_word_evidence(negator, words[0], bool(words[1:])) if words else False


def _english_word_evidence(negator: str, head: str, followed: bool) -> bool | None:
    """How ``head``, the first word an English ``negator`` negates that names something, reads; ``followed`` says
    whether more words come after it.  True: a name (identifier marks, camelCase, a capital on a word no clear
    ending or list makes a description and no open reading leaves unresolved) or a verb of liking, using or doing
    aimed at an object.  False: a description, what the subject cannot do, a past that did not happen, a verb of
    having or of what a thing does, a listed adjective, or a clear adjective or participle ending; capitals do not
    make such a word a name.  None: neither.  An open reading (ible, ant, or default) is unresolved in either
    case, so a capital does not deny it and a value the slot holds or held still can."""
    word = head.casefold()
    lemmas = _lemmas(word)
    if _EN_UNABLE.search(negator):
        return followed and bool(lemmas & {"stand", "bear"})
    if _EN_DID_NOT.search(negator) or lemmas & _EN_HAVING:
        return False
    if _written_as_a_name(head):
        return True
    if followed and lemmas & _EN_REJECTING and not word.endswith("ed"):
        return True
    stem = word.rsplit("-", 1)[-1]
    if word in _EN_STATE_WORDS or _EN_STATE_ENDING.search(stem):
        return False
    open_reading = word in _EN_OPEN_WORDS or _EN_OPEN_ENDING.search(stem) is not None
    if (head[:1].isupper() and not open_reading) or (
        followed and bool(lemmas & _EN_REJECTING or _EN_VERBAL.search(negator))
    ):
        return True
    return None


def _target_evidence(negation: re.Match, target: str) -> bool | None:
    """What a negation's own text says of the ``target`` it negates, slot history aside.

    True: it names an alternative or directs liking, using or doing at one: a complete number-word quantity, a
    Chinese verb of choice, Latin text, an identifier or a quotation after 不是 or 并非, an aversion, an English name
    or action (``_english_word_evidence``).  False: it describes its subject: 没, 未, 尚未 and 未曾, 不 before a
    verb of being somewhere, having or opening, 能 or a 会 without 再, a phrase after 不是 or 并非 that ends in 的 or
    opens with a degree word, an English description.  None: the text decides neither (不是主节点, \"not bash\", an
    open reading such as \"not Default\" or \"not Ansible\").
    """
    chinese = negation.group("negator") is not None or negation.group("outright") is not None
    verb = negation.group("verb")
    if _NAMED_QUANTITY.fullmatch(target.removeprefix("是").strip()):
        return True
    if chinese:
        if negation.group("negator") not in (None, "不", "别"):
            return False
        modal = negation.group("modal") or ""
        if verb in _ZH_STATE_VERBS or "能" in modal or re.search("会(?!再)", modal):
            return False
        if verb == "是" or negation.group("outright") in {"并非", "绝非"}:
            description = target.removeprefix("是").strip()
            if description.endswith("的") or _ZH_DEGREE.match(description):
                return False
            return True if re.search(r"[A-Za-z0-9#@./_]|[\"'“‘「『]", description) else None
        return True
    if negation.group("dislike") or re.match(r"\s*[\"'“‘「『].+?[\"'”’」』]", target):
        return True
    return _english_evidence(negation.group(), target)


def read_denial(payload, held_values=()) -> Denial | None:
    """The denial a fact's or a preference's value is, or None: read from its own text first (``_target_evidence``),
    then from the values its slot holds or held (``held_values``) where the text decides nothing.

    A value whose text describes its subject is the slot's value even when it negates the value held: 不在线 after
    在线, \"not online\" after online, \"not Configured\" after configured, 不是默认的 after 默认的 each say the state
    changed.  History settles only what the text leaves open: \"not bash\" or 不是主节点 denies bash or 主节点 where
    the slot holds or held it, and is a value where it never did.  An open reading stays unresolved in either case, so
    \"not Ansible\", \"not vagrant\" and \"not Default\" are not names by their capitals, and a held Ansible, vagrant
    or default is denied.  Descriptions form an open set, so an unrecognized
    one stays a value rather than withdraw stored data; that ambiguity is the accepted cost.  A value that goes on to
    say what holds instead (\"not X but Y\") is a denial whatever its target.  Fixed expressions and polar answers
    remain values; capitals throughout supply no evidence without matching history.  Target comparison uses the
    same normalization as ``negates``; a contrast and trailing time words belong to the construction, not the target.
    """
    negation = _negation(payload)
    if negation is None:
        return None
    value = str(payload.get("value_text") or "")
    rest = value[negation.end() :]
    contrast = _CONTRAST.search(rest)
    target = _SINCE.sub("", rest if contrast is None else rest[: contrast.start()])
    read = Denial(_value_words(target), contrast is not None)
    if not read.names_a_value:
        return None
    evidence = _target_evidence(negation, target)
    if evidence is False and contrast is None:
        return None
    if any(_target_negates(read, held) for held in held_values):
        return read
    if _set_in_capitals(value, negation.group()):
        return None
    return read if contrast is not None or evidence else None


def is_denial(payload, held_values=()) -> bool:
    """Whether a claim's value says what its slot is not (``read_denial``): "not CircleCI", "不吃辣", "don't like
    PowerShell"; not 不错, 没问题, 不一定喜欢 X, "not bad", "isn't it X?", NOT NULL, NOT_STARTED, 未开始, 不在线 or
    "not started", which say what state their subject is in."""
    return read_denial(payload, held_values) is not None


def negates(denial, held) -> bool:
    """Whether the denial ``denial`` says that the claim ``held`` (a payload) does not hold.

    The target the denial denies (``read_denial``) and ``held``'s value are compared as the store compares values
    (NFKC, case-folded), word by word: a CJK character is a word, an identifier is one word with its marks (v3.1,
    #ops-pager), a Latin word is taken by the stem the fact gate gives a predicate's words, and an article is no word.
    The denial negates the held value when the held value holds all the target after a copula (不是周三 negates
    周三下午, 不吃辣 negates 爱吃辣, "don't drink coffee" negates "drinks coffee"), or when the target is the held
    value itself under the verbs of the construction ("don't like PowerShell" negates PowerShell, 不喜欢喝红茶 negates
    红茶).  A longer target does not negate a shorter value: its other words narrow what it denies, so 不是周三下午
    leaves 周三 standing, as "not v3" leaves v3.1, "not 4 GB" 14 GB and 不喜欢喝红茶 茶.  A target of verbs alone
    (不吃了) negates nothing.
    """
    read = read_denial(denial, (held,))
    return read is not None and _target_negates(read, held)


def _target_negates(read: Denial, held) -> bool:
    """Compare a target with a held assertion, never with another negation of it: "not fast" cannot deny itself."""
    value = str(held.get("value_text") or "")
    if _NEGATOR.match(value):
        return False
    holds = _value_words(value)
    if not read.names_a_value or not holds:
        return False
    denied = read.target
    named = denied[1:] if denied[0] in _COPULA and len(denied) > 1 else denied
    lead = len(denied) - len(holds)
    return _runs_within(holds, named) or (
        lead > 0 and denied[lead:] == holds and all(word in _DENYING_VERBS for word in denied[:lead])
    )


def _runs_within(words: tuple[str, ...], run: tuple[str, ...]) -> bool:
    """Whether ``run`` stands in ``words`` as consecutive words."""
    return any(words[at : at + len(run)] == run for at in range(len(words) - len(run) + 1))
