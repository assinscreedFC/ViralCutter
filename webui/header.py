import os
import sys

# Necessary if this file is imported from app.py which is in the same dir but we need root for i18n
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(WORKING_DIR)

from i18n.i18n import I18nAuto
i18n = I18nAuto()

badges = ""

description = f"""
<div style="text-align: center;">

<h1>ViralCutter</h1>
<p style="font-size: 1.1em; margin-bottom: 20px;">{i18n('Bem-vindo ao ViralCutter! A ferramenta definitiva para transformar vídeos longos em clipes virais com o poder da IA.')}</p>

<div style="display: inline-block; text-align: left; background: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
<p style="margin-bottom: 10px;"><strong>{i18n('Aqui você pode:')}</strong></p>
<ul style="margin: 0; padding-left: 20px;">
<li>✂️ <strong>{i18n('Cortes Automáticos')}</strong>: {i18n('Identifique e corte os melhores momentos baseado em viralidade.')}</li>
<li>📝 <strong>{i18n('Legendas Dinâmicas')}</strong>: {i18n('Crie legendas estéticas (Estilo Hormozi) automaticamente.')}</li>
<li>🤖 <strong>{i18n('IA Avançada')}</strong>: {i18n('Suporte integrado para')} <strong>Gemini</strong> e <strong>G4F</strong>.</li>
<li>📱 <strong>{i18n('Foco em Vertical')}</strong>: {i18n('Detecção facial inteligente para vídeos verticais (TikTok/Shorts/Reels).')}</li>
</ul>
</div>

</div>
"""