"""Bounded, source-time zone snapshots; never infer history from forecasts."""
from collections import deque
import pandas as pd
from backend.geo import aggregate


class ZoneHistory:
    def __init__(self):
        self.snapshots = deque()
        self.last = None

    def capture(self, engine):
        now = engine.clock
        if now is None or self.last is not None and (now-self.last).total_seconds() < 30:
            return
        self.last = now
        modes = {mode: {c['cell_id']: c for c in aggregate(list(engine.vehicles.values()), now, mode)}
                 for mode in ('current', 'forecast')}
        self.snapshots.append((now, modes))
        while self.snapshots and (now-self.snapshots[0][0]).total_seconds() > 1800:
            self.snapshots.popleft()

    def rows(self, cell_id, mode):
        return [dict(timestamp=str(at), **modes[mode][cell_id]) for at, modes in self.snapshots
                if cell_id in modes[mode]]

    def decorate(self, cells, now, mode):
        for cell in cells:
            cutoff = now-pd.Timedelta(minutes=5)
            reference = next(((at, modes[mode].get(cell['cell_id'])) for at, modes in reversed(self.snapshots)
                              if at <= cutoff), None)
            previous = reference[1] if reference and (cutoff-reference[0]).total_seconds() <= 60 else None
            delta = round(cell['risk_score']-previous['risk_score'], 1) if previous else None
            cell.update(risk_trend=delta, trend_definition='risk now minus risk 5 minutes ago',
                        trend='unknown' if delta is None else 'worsening' if delta > 3 else 'improving' if delta < -3 else 'stable')
            recommendations = []
            if cell['risk_factors']['speed_drop'] >= 3:
                recommendations.append('Проверить дорожную ситуацию: зафиксировано снижение скорости.')
            if cell['problem_vehicles_count'] >= 2:
                recommendations.append('Проверить общий участок: задержка затрагивает несколько ТС в зоне.')
            if (cell['predicted_delay'] or 0) >= 300:
                recommendations.append('Проверить ожидаемое опоздание на ближайшее целевое событие.')
            cell['recommendations'] = recommendations or ['Продолжить наблюдение за зоной.']
        return cells
