---
status: awaiting_human_verify
trigger: "Bug A: DeepSeek sort le template verbatim. Bug B: DeepSeek choisit systématiquement les durées minimales (42s)"
created: 2026-03-15T00:00:00Z
updated: 2026-03-15T00:00:00Z
---

## Current Focus

hypothesis: CONFIRMED pour les deux bugs — fixes appliqués
test: Human verification sur prochain run pipeline
expecting: Bug A éliminé — plus de "REPLACE with" ou "Viral Hook Title" dans les réponses. Bug B atténué — end_time_ref respecte le delta minimum.
next_action: Attendre confirmation humaine après prochain run

## Symptoms

expected:
- Bug A: DeepSeek retourne du JSON avec des segments réels extraits du transcript
- Bug B: DeepSeek choisit des durées variées entre 42s et 90s selon le contenu, pas toujours le minimum

actual:
- Bug A: response_raw_part_1.txt contient `"start_text": "You are a a"` et title="Viral Hook Title" (template non rempli)
- Bug B: end_time_ref=2803 avec start_time_ref=2790 (delta=13s seulement), étendu à 42s par process_segments()

errors:
- Bug A: Pas d'erreur — réponse invalide silencieuse parsée comme template
- Bug B: [WARN] Segmento menor que duration min (X.XXs < 42s). Estendendo para 42s.

reproduction:
- Bug A: Vidéo longue avec chunking — chunk 1 retourne souvent le template
- Bug B: N'importe quelle vidéo avec DeepSeek-R1

## Eliminated

- hypothesis: placeholder {transcript_chunk} non remplacé
  evidence: prompt_part_1.txt ligne 117 montre le transcript correctement injecté (1788 lignes de transcript réel)
  timestamp: 2026-03-15

- hypothesis: Bug B causé par end_time_ref absent/non parsé dans process_segments
  evidence: process_segments() reçoit bien end_time_ref mais la valeur retournée par DeepSeek est end=2803, start=2790 (delta=13s) — le LLM choisit des timestamps naturels courts, pas le pipeline
  timestamp: 2026-03-15

## Evidence

- timestamp: 2026-03-15
  checked: response_raw_part_1.txt
  found: |
    <think> bloc (lignes 1-17) : DeepSeek raisonne correctement
    JSON retourné : "start_text": "You are a a" — début du system prompt
    "title": "Viral Hook Title" — placeholder non rempli
    DeepSeek dit dans <think> ligne 6 : "I notice that the user provided an example output with a single segment"
    → Il traite le json_template comme un exemple REMPLI à imiter, pas un schéma vide
  implication: Les valeurs placeholder du json_template ("Viral Hook Title", "Why this is viral?") ressemblent à des exemples réels — DeepSeek les retourne verbatim

- timestamp: 2026-03-15
  checked: response_raw_part_2.txt
  found: start_time_ref=2790, end_time_ref=2803 (delta=13s pour un segment "VIP Music Privilege")
  implication: DeepSeek choisit les timestamps naturels de l'échange (13s réels) — ignore la contrainte min dans le json_template car elle est formulée comme description de champ pas comme règle dure

- timestamp: 2026-03-15
  checked: call_pleiade() ligne 546
  found: tout envoyé en role=user, pas de system message séparé
  implication: facteur aggravant Bug A — DeepSeek-R1 confond instructions et contenu plus facilement sans system prompt dédié

- timestamp: 2026-03-15
  checked: prompt.txt ligne 39 (avant fix)
  found: "Each segment MUST be between {min_duration}s and {max_duration}s" — contrainte générale
  implication: Contrainte présente mais sans exemple numérique ni instruction sur end_time_ref — insuffisant pour DeepSeek-R1 reasoning mode

## Resolution

root_cause: |
  Bug A: Le json_template utilisait des strings placeholder qui ressemblent à des exemples réels ("Viral Hook Title", "Why this is viral?", "Exact first 5-10 words"). DeepSeek-R1 les interprète comme exemples à imiter et les retourne verbatim au lieu d'extraire du transcript.

  Bug B: La contrainte de durée dans le json_template était formulée comme description de champ ("must differ from start_time_ref by 42 to 90 seconds") sans exemple numérique concret. DeepSeek-R1 choisit les timestamps naturels du transcript (souvent 10-20s pour un échange court) et ignore la contrainte.

fix: |
  1. scripts/create_viral_segments.py — json_template reécrit :
     - Placeholders changés de "Viral Hook Title" → "REPLACE with viral title in transcript language"
     - start_time_ref et end_time_ref donnent maintenant des valeurs exemple (120, 185) au lieu de strings descriptives
     - Bloc IMPORTANT RULES ajouté après le JSON avec règle explicite : "end_time_ref MUST be start_time_ref + 42 to 90. NEVER return end_time_ref ≈ start_time_ref"

  2. prompt.txt — Rule 4 renforcée :
     - Ajout ligne CRITICAL avec exemple numérique (start=100 → end=142 à 190)
     - Exemples BAD/GOOD explicites (start=500,end=513 WRONG vs start=500,end=565 preferred)
     - Ajout REMINDER juste avant {json_template} rappelant les deux contraintes critiques

verification: En attente du prochain run pipeline
files_changed:
  - scripts/create_viral_segments.py (json_template lignes 920-948)
  - prompt.txt (rule 4 + section OUTPUT FORMAT)
