Проверь реестр утверждений против переданных источников.
Верни JSON: {"issues": [{"claim_id", "type", "severity", "hint"}]}.
Типы: ungrounded, unjustified_inference, contradiction, misleading, overclaim,
layer_violation, incomplete, impact_inconsistent, paraphrase_only,
single_factor, no_consequence, news_only_risks, impact_rationale_boilerplate.
severity: minor или blocking.
Не предлагай новый текст инсайта и не усредняй противоречия.

Дополнительно отмечай:
- paraphrase_only (blocking): model в why_it_matters ≈ fact/title по overlap токенов,
  без ответа «что это значит для компании»;
- single_factor (blocking): len(mechanism_ids) ≥ 2, а в draft claims только один mechanism_id;
- no_consequence (blocking): нет ни risks, ни opportunities, ни scenario с логикой последствий
  (пустой logic или заглушка «условный масштаб» не считаются);
- news_only_risks (minor; blocking при impact_level=high): risks ⊆ extraction risks из новости,
  при этом есть profile features и/или calc — нет клиентской привязки;
- impact_rationale_boilerplate (blocking): rationale сводится к «Уровень посчитан по рубрике»
  без факторов профиля/механизмов;
- incomplete: пустые risks/opportunities при сработавших механизмах закрыты только
  «в тексте не названы» (нужны клиентские формулировки или «нейтральный эффект при
  текущих признаках»).
