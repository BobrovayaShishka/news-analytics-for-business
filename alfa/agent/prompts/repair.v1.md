Перепиши только утверждения с замечаниями.
Сохрани id, слой и ссылки, которые есть в источниках.
Число без fact, card или calc удали.
Неподтверждённое переведи в hypothesis или убери.
Не добавляй новые механизмы и не вызывай инструменты: это сделает код.

Если среди issues есть paraphrase_only, single_factor, no_consequence,
news_only_risks или impact_rationale_boilerplate — чини логику последствий,
а не только grounding чисел и ссылок:
- paraphrase_only / single_factor: why_it_matters = событие → признаки/механизм → следствие
  для профиля; при ≥2 mechanism_ids покрой независимые контуры отдельными model-claims;
- no_consequence / news_only_risks: risks и opportunities через профиль/механизмы/calc,
  не копируй extraction risks из новости; сценарии — 2–3 ветки с разной логикой последствий;
- impact_rationale_boilerplate: обоснование с ключевыми факторами, согласованное с рубрикой,
  без голого «Уровень посчитан по рубрике».
