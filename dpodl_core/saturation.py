MIN_DECREASE = -0.001
MAX_INCREASE = 0.0005
WINDOW = 10
SLACK_RATE_DECREASE, SLACK_RATE_INCREASE = 0.3, 0.5
SLACK_RATE_DECREASE, SLACK_RATE_INCREASE = SLACK_RATE_DECREASE * WINDOW, SLACK_RATE_INCREASE * WINDOW


def find_stabilization_point(sequence):
    """
    Identifies the point at which the sequence stops significantly decreasing.

    :param sequence: List of float values (the sequence to analyze)
    :return: The index of the first value where the sequence stops significantly decreasing, or None if not found
    """

    if len(sequence) < 2:
        return float("inf")  # A sequence with less than 2 values can't have a significant decrease

    changes = []  # List to track the relative changes in values

    for i in range(1, len(sequence)):
        change = (sequence[i] - sequence[i - 1]) / abs(sequence[i - 1])
        changes.append(change)

    # Check if consecutive changes are consistently smaller than the threshold
    for i in range(WINDOW, len(changes)):
        # Consider the last few changes for oscillation detection
        # check for increasing behaviour:
        if [change > MAX_INCREASE for change in changes[i - WINDOW: i]].count(True) >= SLACK_RATE_INCREASE:
            return i - WINDOW

        # check for decreasing behaviour:
        if [change > MIN_DECREASE for change in changes[i - WINDOW: i]].count(True) >= SLACK_RATE_DECREASE:
            return i

    return float("inf")  # No stabilization point found
