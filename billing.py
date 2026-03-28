import math

RATES_PER_HOUR = {
    "2W": 20.0,
    "4W": 50.0,
    "HEAVY": 90.0,
}


def calculate_amount(vehicle_type: str, duration_minutes: int) -> float:
    hourly_rate = RATES_PER_HOUR[vehicle_type]
    billable_hours = max(1, math.ceil(duration_minutes / 60))
    return round(hourly_rate * billable_hours, 2)
