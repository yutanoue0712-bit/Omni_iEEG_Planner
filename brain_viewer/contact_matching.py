"""Patient-local name matching without guessing electrode identities or numbers."""
from collections import defaultdict
from dataclasses import dataclass
import re
import unicodedata


def normalized_contact_name(name):
    name=unicodedata.normalize('NFKC',str(name)).strip().casefold()
    name=name.translate(str.maketrans({dash:'-' for dash in '−–—‐‑'}))
    return re.sub(r'\s*-\s*','-',name)


def edf_contact_name(name,group):
    """Keep the electrode name literal; compact only its numeric contact suffix."""
    prefix=f'{group}-'
    if isinstance(name,str) and name.startswith(prefix):
        number=name[len(prefix):]
        if number.isascii() and number.isdecimal(): return f'{group}{int(number)}'
    return name


@dataclass(frozen=True)
class NameMatch:
    ids: tuple[str,...] = ()
    issue: str = ''


class ContactNameIndex:
    def __init__(self,contacts):
        self.names=defaultdict(set); self.numbered=defaultdict(set); self.groups=set()
        for contact in contacts:
            name=normalized_contact_name(contact.name); group=normalized_contact_name(contact.group)
            self.names[name].add(contact.uid)
            if not group: continue
            self.groups.add(group)
            number=self.number(name,group)
            if number is not None: self.numbered[group,number].add(contact.uid)

    @staticmethod
    def number(name,group):
        if name.startswith(group):
            match=re.fullmatch(r'-?([0-9]+)',name[len(group):])
            if match: return int(match[1])
        return None

    def contacts_for(self,name):
        name=normalized_contact_name(name)
        ids=set(self.names.get(name,()))
        for group in self.groups:
            number=self.number(name,group)
            if number is not None: ids.update(self.numbered.get((group,number),()))
        return ids

    def arity(self,name):
        if self.contacts_for(name): return 1
        return 2 if '-' in normalized_contact_name(name) else 1

    def match(self,name):
        """Both bipolar endpoints must exist uniquely; retain their written order.

        Enumerating separators also permits hyphens in actual contact names.
        Number gaps, truncated labels and duplicate names never imply a match.
        """
        name=normalized_contact_name(name)
        candidates={(uid,) for uid in self.contacts_for(name)}
        partial=False; same=False; ambiguous=False
        for i,char in enumerate(name):
            if char!='-': continue
            first=self.contacts_for(name[:i]); second=self.contacts_for(name[i+1:])
            partial=partial or bool(first) != bool(second)
            if first and second and (len(first)>1 or len(second)>1): ambiguous=True
            for a in first:
                for b in second:
                    if a==b: same=True
                    else: candidates.add((a,b))
        if ambiguous or len(candidates)>1: return NameMatch(issue='ambiguous')
        if len(candidates)==1: return NameMatch(next(iter(candidates)))
        return NameMatch(issue='same_contact' if same else 'missing' if partial else 'unmatched')
