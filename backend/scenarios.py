"""Deterministic dispatcher scenarios. User assumptions, not predictions or causality."""
from pydantic import BaseModel, ConfigDict, Field


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    cycle_minutes: float = Field(ge=1, le=600)
    fleet: int = Field(ge=1, le=1000)
    reserve: int = Field(ge=0, le=100)
    segment_km: float = Field(ge=0, le=200)
    baseline_speed_kmh: float = Field(gt=0, le=130)
    scenario_speed_kmh: float = Field(gt=0, le=130)
    added_dwell_seconds: float = Field(ge=0, le=3600)
    horizon_minutes: int = Field(ge=10, le=15)
    dispatch_minutes: float = Field(ge=0, le=180)


def calculate(s: Scenario):
    before = 3600 * s.segment_km / s.baseline_speed_kmh
    after = 3600 * s.segment_km / s.scenario_speed_kmh + s.added_dwell_seconds
    delta = after - before
    cycle = s.cycle_minutes + delta / 60
    if cycle <= 0:
        raise ValueError("Scenario produces a non-positive cycle time")
    effective_reserve = s.reserve if s.dispatch_minutes <= s.horizon_minutes else 0
    headway = s.cycle_minutes / s.fleet
    new_headway = cycle / (s.fleet + effective_reserve)
    return {"kind": "deterministic_scenario", "inputs": s.model_dump(),
            "segment_before_seconds": before, "segment_after_seconds": after, "delta_seconds": delta,
            "cycle_after_minutes": cycle, "headway_before_minutes": headway,
            "headway_after_minutes": new_headway, "wait_before_minutes": headway/2,
            "wait_after_minutes": new_headway/2, "effective_reserve": effective_reserve,
            "formulas": ["t₀ = 3600·L / v₀", "t₁ = 3600·L / v₁ + dwell", "Δt = t₁ − t₀",
                         "C₁ = C₀ + Δt/60", "H₀ = C₀/N", "H₁ = C₁/(N + R_effective)", "E[W] = H/2"],
            "assumptions": ["Все входы введены пользователем; скорости не оценены по погоде или ДТП",
                            "Изменённый участок проходится один раз за оборот; остальные участки неизменны",
                            "Равномерное движение, одинаковые ТС, случайное прибытие пассажиров",
                            "Резерв учитывается только при времени выпуска не больше горизонта",
                            "Интервал после изменения — стационарная оценка, не прогноз переходного процесса",
                            "Результат не является ML-прогнозом и не меняет расписание"]}
