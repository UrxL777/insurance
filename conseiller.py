# AGENT CONSEILLER
# Un agent qui conseille l'assuré pour payer moins cher et prendre soin de sa santé.
# 1. Il reçoit le profil de l'assuré et sa charge estimée.
# 2. Il utilise le modèle de prédiction comme OUTIL : il simule ce que deviendrait la charge
#    si l'assuré changeait ce qui dépend de lui (le tabac, l'IMC).
# 3. Il donne ses conseils à partir de ces chiffres, et garde la conversation en mémoire.

import os
import re
import pandas as pd
from openai import OpenAI

MODELE = "openai/gpt-oss-20b"        # modèle principal (Groq)
MODELE_SECOURS = "qwen/qwen3.8-27b"  # si le modèle principal est indisponible
IMC_CIBLE = 25                       # limite haute d'un IMC normal selon l'OMS

client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")

NOMS_REGION = {0: "Nord-Est", 1: "Nord-Ouest", 2: "Sud-Est", 3: "Sud-Ouest"}


def decrire_profil(profil):
    """Le profil de l'assuré en mots simples."""
    return (f"{profil['age']} ans, {'homme' if profil['genre'] == 1 else 'femme'}, IMC {profil['imc']}, "
            f"{profil['enfants']} enfant(s), {'fumeur' if profil['fumeur'] == 1 else 'non-fumeur'}, "
            f"région {NOMS_REGION.get(profil['region'], profil['region'])}")


def simuler(modele, colonnes, profil):
    """
    L'OUTIL de l'agent : le modèle recalcule la charge en changeant seulement
    ce que l'assuré peut changer lui-même (le tabac et l'IMC).
    """
    def predire(p):
        return round(float(modele.predict(pd.DataFrame([[p[c] for c in colonnes]], columns=colonnes))[0]))

    scenarios = [{"situation": "situation actuelle", "charge": predire(profil)}]
    if profil["fumeur"] == 1:
        scenarios.append({"situation": "s'il arrête de fumer", "charge": predire({**profil, "fumeur": 0})})
    if profil["imc"] > IMC_CIBLE:
        scenarios.append({"situation": f"si son IMC descend à {IMC_CIBLE}", "charge": predire({**profil, "imc": IMC_CIBLE})})
    if profil["fumeur"] == 1 and profil["imc"] > IMC_CIBLE:
        scenarios.append({"situation": f"s'il arrête de fumer et que son IMC descend à {IMC_CIBLE}",
                          "charge": predire({**profil, "fumeur": 0, "imc": IMC_CIBLE})})
    return scenarios


def consigne(profil, scenarios):
    """La consigne de l'agent, avec le profil et les résultats de l'outil."""
    lignes = "\n".join(f"- {s['situation']} : {s['charge']} $ par an" for s in scenarios)
    return f"""Tu es le conseiller de l'application de prédiction des charges d'assurance santé. Tu aides l'assuré à payer moins cher et à prendre soin de sa santé, avec bienveillance, comme un bon agent d'assurance.

Profil de l'assuré : {decrire_profil(profil)}.
Charges estimées par le modèle (calculées par l'application, chiffres exacts) :
{lignes}

Ta mission :
- Explique simplement ce qui fait monter sa charge et ce qu'il peut changer lui-même, en t'appuyant sur les chiffres ci-dessus.
- Donne des conseils de prévention concrets et réalistes (arrêt du tabac, activité physique, alimentation, sommeil, bilans de santé réguliers), adaptés à son profil.
- Ne conseille jamais de changer ce qui ne dépend pas de lui (âge, sexe, région, nombre d'enfants).

Tes limites :
- Utilise seulement les montants de la liste ci-dessus ; n'invente aucun autre montant.
- Dis que ce sont des estimations du modèle, pas un prix garanti par un assureur.
- Tu ne fais pas de diagnostic médical : pour une question sur sa santé personnelle, oriente vers un médecin.
- Si la question n'a rien à voir avec l'assurance santé ou la prévention, réponds gentiment que tu es là pour ces sujets.

Style : en français, vouvoiement, phrases courtes, 4 à 6 phrases ou une courte liste. Pas de tableau, pas d'émoji."""


def _montants_inventes(texte, scenarios):
    """Les montants en $ écrits par l'agent qui ne viennent pas de l'outil."""
    autorises = {s["charge"] for s in scenarios}
    autorises |= {abs(a - b) for a in autorises for b in autorises}  # les économies
    trouves = re.findall(r"(\d[\d\s  .,]*)\s?\$", texte)
    faux = []
    for brut in trouves:
        chiffres = re.sub(r"[^\d]", "", brut.split(",")[0])
        if chiffres and not any(abs(int(chiffres) - a) <= 1 for a in autorises):
            faux.append(brut.strip())
    return faux


def _appeler(messages):
    """Appelle le modèle principal, ou le modèle de secours si Groq le refuse."""
    for modele in (MODELE, MODELE_SECOURS):
        try:
            reponse = client.chat.completions.create(model=modele, messages=messages, max_tokens=900, temperature=0.4)
            texte = reponse.choices[0].message.content or ""
            return re.sub(r"<think>.*?</think>", "", texte, flags=re.DOTALL).strip()
        except Exception as e:
            print(f"[CONSEILLER] {modele} indisponible : {str(e)[:150]}")
    return None


def repondre(historique, profil, scenarios):
    """
    Répond au dernier message de l'assuré. historique = [{"role": "user"|"assistant", "content": ...}]
    Si un montant n'est pas dans les résultats de l'outil, l'agent réécrit une fois.
    """
    messages = [{"role": "system", "content": consigne(profil, scenarios)}] + historique[-10:]
    texte = _appeler(messages)
    if texte is None:
        return "Le conseiller n'est pas disponible pour le moment. Réessayez dans quelques instants."
    faux = _montants_inventes(texte, scenarios)
    if faux:
        print(f"[CONSEILLER] Montants absents des simulations : {faux} → réécriture")
        nouveau = _appeler(messages + [{"role": "assistant", "content": texte},
                                       {"role": "user", "content": "Réécris ta réponse en utilisant seulement les montants de la liste fournie."}])
        if nouveau and not _montants_inventes(nouveau, scenarios):
            texte = nouveau
        else:
            texte = re.sub(r"\d[\d\s  .,]*\s?\$", "", texte)  # dernier filet : montants retirés
    return texte
