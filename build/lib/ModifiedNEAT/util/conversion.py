
from numba import njit, float64, int32, typeof

STR = typeof('str')


@njit
def float_to_str(value: float, zeros_lim=2, dec_lim=8):
    if value == 0:
        return '0.0'

    # Handle negative numbers
    if value < 0:
        return '-' + float_to_str(-value, zeros_lim, dec_lim)

    # Separate the integer and decimal parts
    int_part = int(value)
    dec_part = value - int_part

    # Convert integer part to string
    int_str = ''
    while int_part > 0:
        int_str = chr(ord('0') + int_part % 10) + int_str
        int_part //= 10

    # Add leading zero if necessary
    if int_str == '':
        int_str = '0'

    # Convert decimal part to string with truncation conditions
    dec_str = ''
    zeros_count = 0
    total_decimals = 0

    while dec_part > 0 and zeros_count < zeros_lim and total_decimals < dec_lim:
        dec_part *= 10
        digit = int(dec_part)
        dec_str += chr(ord('0') + digit)
        dec_part -= digit

        if digit == 0:
            zeros_count += 1
        else:
            zeros_count = 0

        total_decimals += 1

    # Truncate trailing zeros if necessary
    if zeros_count >= zeros_lim or total_decimals >= dec_lim:
        dec_str = dec_str.rstrip('0')

    return int_str + '.' + dec_str if dec_str else int_str + '.0'
