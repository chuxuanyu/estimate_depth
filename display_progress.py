from datetime import datetime
import sys
from time import time
from collections import deque
def progress_bar_str(percentage, bar_length=20, bar_marker='=', show_bar=True):
    if percentage < 0:
        raise ValueError("percentage is not in the range [0, 1]")
    elif percentage > 1:
        percentage = 1
    if not isinstance(bar_length, int) or bar_length < 1:
        raise ValueError("bar_length must be an integer >= 1")
    if not isinstance(bar_marker, str) or len(bar_marker) != 1:
        raise ValueError("bar_marker must be a string of length 1")
    # generate output string
    if show_bar:
        str_param = "[%-" + str(bar_length) + "s] %d%%"
        bar_percentage = int(percentage * bar_length)
        return str_param % (bar_marker * bar_percentage, percentage * 100)
    else:
        return "%d%%" % (percentage * 100)


def print_dynamic(str_to_print):
    sys.stdout.write("\r{}".format(str_to_print.ljust(80)))
    sys.stdout.flush()

def print_progress(iterable, prefix='',
                   show_bar=True, show_count=True, show_eta=True,
                   end_with_newline=True, min_seconds_between_updates=0.1):

    if prefix != '':
        prefix += ': '
        bar_length = 10
    else:
        bar_length = 20
    n =  len(iterable)

    timings = deque([], 100)
    time1 = time()
    last_update_time = 0
    for i, x in enumerate(iterable):
        yield x
        time2 = time()
        timings.append(time2 - time1)
        time1 = time2
        remaining = n - i
        if time2 - last_update_time < min_seconds_between_updates:
            continue
        last_update_time = time2
        duration = datetime.utcfromtimestamp(sum(timings) / len(timings) *
                                             remaining)
        bar_str = progress_bar_str(i / n, bar_length=bar_length,
                                   show_bar=show_bar)
        count_str = ' ({}/{})'.format(i, n) if show_count else ''
        eta_str = (" - {} remaining".format(duration.strftime('%H:%M:%S'))
                   if show_eta else '')
        print_dynamic('{}{}{}{}'.format(prefix, bar_str, count_str, eta_str))

    # the iterable has now finished - to make it clear redraw the progress with
    # a done message. We also hide the eta at this stage.
    count_str = ' ({}/{})'.format(n, n) if show_count else ''
    bar_str = progress_bar_str(1, bar_length=bar_length, show_bar=show_bar)
    print_dynamic('{}{}{} - done.'.format(prefix, bar_str, count_str))

    if end_with_newline:
        print('')
