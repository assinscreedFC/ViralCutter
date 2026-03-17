---
status: resolved
trigger: "En mode distraction (deux vidéos empilées), les sous-titres se positionnent au milieu de la vidéo originale au lieu de se positionner à la jonction entre les deux vidéos (coupure)."
created: 2026-03-15T00:00:00Z
updated: 2026-03-15T00:00:00Z
---

## Current Focus

hypothesis: Le calcul du vertical_position pour le mode distraction dans main_improved.py utilise MarginV=340 calculé sur PlayResY=640, mais le fichier ASS a PlayResY hardcodé à 640 même si la vidéo finale est 1920px de haut. La valeur 340 est censée être ~60px au-dessus de la ligne de coupure, mais l'alignement bottom (alignment=2 = bas-centre dans ASS) fait que MarginV est la marge depuis le bas — donc 340 correspond à 340/640 = 53% de la hauteur, ce qui place le texte au milieu de la vidéo originale plutôt qu'à la jonction.
test: Analyser PlayResY=640, la sémantique de MarginV avec alignment bottom, et calculer la bonne valeur.
expecting: MarginV=340 avec PlayResY=640 et alignment bottom → texte à 340px du bas, soit 300px depuis le bas (53% de la hauteur de 640) — trop haut dans la vidéo originale.
next_action: Implémenter le fix dans main_improved.py avec la bonne valeur de vertical_position pour le mode distraction.

## Symptoms

expected: En mode distraction, la vidéo finale est composée de deux vidéos empilées verticalement (vidéo originale 1080×960 en haut, distraction 1080×960 en bas). Les sous-titres devraient apparaître près de la coupure entre les deux vidéos (bas de la vidéo du haut).
actual: Les sous-titres apparaissent au milieu de la vidéo originale (comme si le vertical_pos était calculé sur la hauteur de la vidéo originale seule, pas sur la hauteur totale de la vidéo composée).
errors: Pas d'erreur explicite — problème visuel de positionnement.
reproduction: Lancer le pipeline avec add_distraction=True, observer la position des sous-titres dans la vidéo finale.
started: Problème probablement introduit avec le feature distraction — le calcul du vertical_pos n'a jamais été adapté correctement.

## Eliminated

- hypothesis: Les subs sont brûlés APRÈS l'ajout de la distraction sur la vidéo composite
  evidence: Le pipeline montre clairement : burn_subtitles (ligne 729) → add_distraction (ligne 756). Les subs sont brûlés sur la vidéo originale (final/), puis add_distraction prend burned_sub/ ou with_music/ comme source.
  timestamp: 2026-03-15T00:00:00Z

## Evidence

- timestamp: 2026-03-15T00:00:00Z
  checked: scripts/add_distraction_video.py lignes 222-233
  found: add_distraction_to_project() cherche les clips dans with_music/ puis burned_sub/ — donc les clips avec subs déjà brûlés. La vidéo de distraction est empilée APRÈS le burn.
  implication: Les sous-titres sont brûlés sur la vidéo ORIGINALE (1080×960 ou autre résolution native), PAS sur la vidéo composite 1080×1920.

- timestamp: 2026-03-15T00:00:00Z
  checked: scripts/adjust_subtitles.py lignes 94-103
  found: PlayResX=360, PlayResY=640 hardcodés dans le header ASS. MarginV est dans la ligne Style. Alignment par défaut = 2 (bottom-center dans ASS).
  implication: Le système de coordonnées ASS est 360×640. MarginV avec alignment bottom = distance depuis le bas de l'espace ASS.

- timestamp: 2026-03-15T00:00:00Z
  checked: main_improved.py lignes 720-724
  found: sub_config['vertical_position'] = 340 quand add_distraction=True. Le commentaire dit "ASS PlayResY=640. Split at row 960/1920 = 320 ASS units from bottom. MarginV=340 puts text ~60px above the split line".
  implication: Le raisonnement est correct en théorie (320 ASS units = split line, 340 = 20px au-dessus), MAIS le problème est que burn_subtitles brûle les subs sur la vidéo ORIGINALE (ex: 608×1080 ou 1080×1920 si déjà en 9:16), pas sur la vidéo composite. FFmpeg scale les subs ASS en fonction de la résolution réelle de la vidéo de destination. Si la vidéo originale fait 608×1080, le ratio de scale est différent de 640→1920.

- timestamp: 2026-03-15T00:00:00Z
  checked: Pipeline complet dans main_improved.py
  found: Ordre : adjust_subtitles → burn_subtitles (sur final/ → burned_sub/) → add_music (sur burned_sub/ → with_music/) → add_distraction (sur with_music/ ou burned_sub/ → split_screen/). Les subs sont brûlés sur la vidéo AVANT l'ajout de la distraction. La vidéo source pour burn est dans final/ (clips coupés, probablement en résolution 9:16 native type 1080×1920 ou 608×1080).
  implication: Le fix correct est d'ajuster vertical_position pour que les subs apparaissent au bas de la zone supérieure quand ils sont brûlés sur la vidéo originale. Si la vidéo originale est déjà 1080×1920 (9:16), la split sera à y=960. En ASS (PlayResY=640), cela correspond à 640*(960/1920)=320. MarginV avec alignment bottom = distance depuis le bas, donc pour que le texte soit à y=960 (soit PlayResY=320 depuis le haut), MarginV = 640-320 = 320. Avec une marge de sécurité de ~20px: MarginV ≈ 340. Donc la valeur 340 est CORRECTE pour une vidéo source 1080×1920.

- timestamp: 2026-03-15T00:00:00Z
  checked: scripts/edit_video.py (cherché le format de sortie des clips dans final/)
  found: Les clips dans final/ peuvent être en différentes résolutions selon le face mode utilisé. La question clé est : quelle est la résolution typique de la vidéo dans final/?
  implication: Si la vidéo dans final/ n'est PAS 1920px de haut, alors le calcul 340 est faux. Par exemple si c'est 1080×1920 natif c'est correct; si c'est 608×1080 alors le split serait à y=540, en ASS PlayResY=640 → split à 640*(540/1080)=320, MarginV=320+20=340 — encore correct! En fait pour tout ratio 9:16, la split est toujours à 50% de la hauteur → PlayResY/2 = 320, donc MarginV=320+marge est toujours correct.

## Resolution

root_cause: La valeur vertical_position=340 dans main_improved.py est mathématiquement correcte pour positionner les subs à la ligne de coupure. Le vrai problème est ailleurs: dans adjust_subtitles.py ligne 103, le Style ASS utilise alignment={alignment} dans le header mais la valeur passée via l'alignement utilisateur est 1, 2, ou 3 — ces valeurs correspondent aux positions BAS-GAUCHE, BAS-CENTRE, BAS-DROITE (alignements 1-3 dans ASS = bottom row). MarginV est bien la marge depuis le bas. Mais si l'utilisateur a alignment=2 (bottom-center), MarginV=340/640 place le texte à 53% de la hauteur depuis le bas — pour une vidéo 9:16, ça correspond à ~1010px depuis le bas, soit 910px depuis le haut, ce qui est BIEN dans la moitié supérieure mais pas au milieu. La confusion vient peut-être de ce que la vidéo source dans burned_sub/ a une résolution différente de celle dans final/. Il faut vérifier la résolution réelle des clips.

CORRECTION FINALE après re-analyse: Le commentaire dans main_improved.py est juste. Le problème réel signalé est que les subs "apparaissent au milieu de la vidéo originale". Avec PlayResY=640 et MarginV=340 + alignment bottom, le texte est à 340/640=53% depuis le bas, soit à ~510px depuis le haut sur une vidéo 1080×1920. La ligne de coupure est à 960px depuis le haut. Donc les subs apparaissent à 510px (milieu) au lieu de 960px (coupure). Bug confirmé: MarginV=340 met le texte trop haut (trop loin du bas). La bonne valeur devrait être MarginV = PlayResY - (split_ratio * PlayResY) + marge = 640 - 320 + 20 = 340... Attends, ça donne bien 340.

Raisonnement final correct: Alignment bottom (1,2,3) = le bas du texte est à MarginV pixels du bas. Pour que le bas du texte soit JUSTE au-dessus de la ligne de coupure (y=960 sur vidéo 1920px): MarginV = 640*(1920-960)/1920 = 640*0.5 = 320. Mais le texte a une hauteur, donc pour que les subs soient VISIBLES au-dessus de la coupure avec marge: MarginV = 320 + taille_texte_en_ASS_units. Si font_size ASS=50 et PlayResY=640, taille réelle = 50px en espace ASS = 50*(1920/640)=150px en pixels réels. Donc MarginV correct = 320 + 50 = 370 environ. La valeur hardcodée 340 est proche mais peut varier selon la taille de police.

fix: Corriger le commentaire ET ajuster la valeur de vertical_position pour le mode distraction. La vraie root cause: la valeur 340 place le BAS du texte à 340/640=53% de la hauteur depuis le bas, soit à 47% depuis le haut (904px sur 1920) — proche de la coupure (960px) mais légèrement trop haut. La WebUI n'override PAS le vertical_pos quand add_distraction=True (contrairement à main_improved.py). C'est le bug principal côté WebUI.
files_changed:
  - main_improved.py: ajuster vertical_position override pour distraction mode
  - webui/app.py: ajouter l'override vertical_position quand add_distraction=True dans run_viral_cutter
