class ParameterValidator:

    @staticmethod
    def min(value, minimum, name):
        if value < minimum:
            raise ValueError(
                f"{name} must be >= {minimum}, got {value}"
            )

    @staticmethod
    def max(value, maximum, name):
        if value > maximum:
            raise ValueError(
                f"{name} must be <= {maximum}, got {value}"
            )

    @staticmethod
    def between(value, minimum, maximum, name):
        if not minimum <= value <= maximum:
            raise ValueError(
                f"{name} must be in [{minimum}, {maximum}], got {value}"
            )

    @staticmethod
    def positive(value, name):
        if value <= 0:
            raise ValueError(
                f"{name} must be > 0, got {value}"
            )

    @staticmethod
    def non_negative(value, name):
        if value < 0:
            raise ValueError(
                f"{name} must be >= 0, got {value}"
            )

    @staticmethod
    def positive_int(value, name):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(
                f"{name} must be a positive integer, got {value!r}."
            )

    @staticmethod
    def int_min(value, minimum, name):
        if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < minimum
        ):
            raise ValueError(
                f"{name} must be an integer >= {minimum}, got {value!r}."
            )



    @staticmethod
    def positive_float(value, name):
        if (
                not isinstance(value, float)
                or isinstance(value, bool)
                or value <= 0
        ):
            raise ValueError(
                f"{name} must be a positive float, got {value!r}."
            )

    @staticmethod
    def float_between(value, minimum, maximum, name):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{name} must be a number, got {value!r}."
            )

        if not minimum <= value <= maximum:
            raise ValueError(
                f"{name} must be in [{minimum}, {maximum}], got {value!r}."
            )

    @staticmethod
    def min_max(minimum, maximum, name):
        if minimum > maximum:
            raise ValueError(f"{name} minimum must be <= maximum, " f"got {minimum} > {maximum}.")

    @staticmethod
    def int_times_leq(value, multiplier, maximum, name):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(
                f"{name} must be an integer, got {value!r}."
            )

        if multiplier * value > maximum:
            raise ValueError(
                f"{multiplier} * {name} must be <= {maximum}, "
                f"got {multiplier * value}."
            )

    @staticmethod
    def int_times_geq(value, multiplier, minimum, name):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(
                f"{name} must be an integer, got {value!r}."
            )

        if multiplier * value < minimum:
            raise ValueError(
                f"{multiplier} * {name} must be >= {minimum}, "
                f"got {multiplier * value}."
            )

    @staticmethod
    def float_times_geq(value, multiplier, minimum, name):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{name} must be a number, got {value!r}."
            )

        if multiplier * value < minimum:
            raise ValueError(
                f"{multiplier} * {name} must be >= {minimum}, "
                f"got {multiplier * value}."
            )
