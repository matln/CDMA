"""
Copyright 2022 Jianchen Li
"""

from pathlib import Path
from rich.theme import Theme
from datetime import datetime
from logging import LogRecord
from rich.logging import RichHandler
from rich.traceback import Traceback
from rich.style import Style, StyleType
from rich.default_styles import DEFAULT_STYLES
from rich.console import Console, ConsoleRenderable
from rich.highlighter import _combine_regex, RegexHighlighter
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from typing import Callable, Iterable, List, Optional, Sequence, Union, TypeVar

ProgressType = TypeVar("ProgressType")

DEFAULT_STYLES.update({"repr.timestamp": Style(color="red", bold=True, italic=False)})
custom_console = Console(theme=Theme(DEFAULT_STYLES))


class MyReprHighlighter(RegexHighlighter):
    """
    Highlights the text typically produced from ``__repr__`` methods.

    Update:
        Add timestmap
    """

    base_style = "repr."
    highlights = [
        r"(?P<tag_start><)(?P<tag_name>[-\w.:|]*)(?P<tag_contents>[\w\W]*?)(?P<tag_end>>)",
        r'(?P<attrib_name>[\w_]{1,50})=(?P<attrib_value>"?[\w_]+"?)?',
        r"(?P<brace>[][{}()])",
        _combine_regex(
            r"(?P<timestamp>[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9]:[0-9][0-9]:[0-9][0-9])",
            r"(?P<ipv4>[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})",
            r"(?P<ipv6>([A-Fa-f0-9]{1,4}::?){1,7}[A-Fa-f0-9]{1,4})",
            r"(?P<eui64>(?:[0-9A-Fa-f]{1,2}-){7}[0-9A-Fa-f]{1,2}|(?:[0-9A-Fa-f]{1,2}:){7}[0-9A-Fa-f]{1,2}|(?:[0-9A-Fa-f]{4}\.){3}[0-9A-Fa-f]{4})",
            r"(?P<eui48>(?:[0-9A-Fa-f]{1,2}-){5}[0-9A-Fa-f]{1,2}|(?:[0-9A-Fa-f]{1,2}:){5}[0-9A-Fa-f]{1,2}|(?:[0-9A-Fa-f]{4}\.){2}[0-9A-Fa-f]{4})",
            r"(?P<uuid>[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12})",
            r"(?P<call>[\w.]*?)\(",
            r"\b(?P<bool_true>True)\b|\b(?P<bool_false>False)\b|\b(?P<none>None)\b",
            r"(?P<ellipsis>\.\.\.)",
            r"(?P<number_complex>(?<!\w)(?:\-?[0-9]+\.?[0-9]*(?:e[-+]?\d+?)?)(?:[-+](?:[0-9]+\.?[0-9]*(?:e[-+]?\d+)?))?j)",
            r"(?P<number>(?<!\w)\-?[0-9]+\.?[0-9]*(e[-+]?\d+?)?\b|0x[0-9a-fA-F]*)",
            r"(?P<path>\B(/[-\w._+]+)*\/)(?P<filename>[-\w._+]*)?",
            r"(?<![\\\w])(?P<str>b?'''.*?(?<!\\)'''|b?'.*?(?<!\\)'|b?\"\"\".*?(?<!\\)\"\"\"|b?\".*?(?<!\\)\")",
            r"(?P<url>(file|https|http|ws|wss)://[-0-9a-zA-Z$_+!`(),.?/;:&=%#]*)",
        ),
    ]


class MyRichHandler(RichHandler):
    def render(
        self,
        *,
        record: LogRecord,
        traceback: Optional[Traceback],
        message_renderable: "ConsoleRenderable",
    ):
        """Render log for display.
        Args:
            record (LogRecord): logging Record.
            traceback (Optional[Traceback]): Traceback instance or None for no Traceback.
            message_renderable (ConsoleRenderable): Renderable (typically Text) containing log message contents.
        Returns:
            ConsoleRenderable: Renderable to display log.
        """
        # path = Path(record.pathname).name + ":" + record.funcName
        path = Path(record.pathname).name
        lineno = str(record.lineno) + f" - {record.funcName}"
        level = self.get_level_text(record)
        time_format = None if self.formatter is None else self.formatter.datefmt
        log_time = datetime.fromtimestamp(record.created)

        log_renderable = self._log_render(
            self.console,
            [message_renderable] if not traceback else [message_renderable, traceback],
            log_time=log_time,
            time_format=time_format,
            level=level,
            path=path,
            line_no=lineno,
            link_path=record.pathname if self.enable_link_path else None,
        )
        return log_renderable


"""
Example:
from rich.progress import Progress
from speakernet.utils.rich_utils import progress_columns

with Progress(*progress_columns) as progress:
    task = progress.add_task(
        "extracting...", total=len(wav_paths)
    )
    while True:
        if counter.value > 0:
            progress.update(task, completed=counter.value)
        if counter.value == len(wav_paths):
            break
"""
progress_columns = (
    SpinnerColumn(),
    "[progress.description]{task.description}",
    TaskProgressColumn(),
    BarColumn(),
    MofNCompleteColumn(),
    "\u2744",
    TimeElapsedColumn(),
    "\u2728",
    TimeRemainingColumn(),
)


def track(
    sequence: Union[Sequence[ProgressType], Iterable[ProgressType]],
    description: str = "Working...",
    total: Optional[float] = None,
    auto_refresh: bool = True,
    console: Optional[Console] = None,
    transient: bool = False,
    get_time: Optional[Callable[[], float]] = None,
    refresh_per_second: float = 10,
    style: StyleType = "bar.back",
    complete_style: StyleType = "bar.complete",
    finished_style: StyleType = "bar.finished",
    pulse_style: StyleType = "bar.pulse",
    update_period: float = 0.1,
    disable: bool = False,
    show_speed: bool = True,
) -> Iterable[ProgressType]:
    """Track progress by iterating over a sequence.

    Args:
        sequence (Iterable[ProgressType]): A sequence (must support "len") you wish to iterate over.
        description (str, optional): Description of task show next to progress bar. Defaults to "Working".
        total: (float, optional): Total number of steps. Default is len(sequence).
        auto_refresh (bool, optional): Automatic refresh, disable to force a refresh after each iteration. Default is True.
        transient: (bool, optional): Clear the progress on exit. Defaults to False.
        console (Console, optional): Console to write to. Default creates internal Console instance.
        refresh_per_second (float): Number of times per second to refresh the progress information. Defaults to 10.
        style (StyleType, optional): Style for the bar background. Defaults to "bar.back".
        complete_style (StyleType, optional): Style for the completed bar. Defaults to "bar.complete".
        finished_style (StyleType, optional): Style for a finished bar. Defaults to "bar.finished".
        pulse_style (StyleType, optional): Style for pulsing bars. Defaults to "bar.pulse".
        update_period (float, optional): Minimum time (in seconds) between calls to update(). Defaults to 0.1.
        disable (bool, optional): Disable display of progress.
        show_speed (bool, optional): Show speed if total isn't known. Defaults to True.
    Returns:
        Iterable[ProgressType]: An iterable of the values in the sequence.

    """

    columns: List["ProgressColumn"] = ([SpinnerColumn()])
    if description:
        columns.append(TextColumn("[progress.description]{task.description}"))
    columns.extend(
        (
            TaskProgressColumn(show_speed=show_speed),
            BarColumn(
                style=style,
                complete_style=complete_style,
                finished_style=finished_style,
                pulse_style=pulse_style,
            ),
            MofNCompleteColumn(),
            "\u2744",
            TimeElapsedColumn(),
            "\u2728",
            TimeRemainingColumn(),
        )
    )

    progress = Progress(
        *columns,
        auto_refresh=auto_refresh,
        console=console,
        transient=transient,
        get_time=get_time,
        refresh_per_second=refresh_per_second or 10,
        disable=disable,
    )

    with progress:
        yield from progress.track(
            sequence, total=total, description=description, update_period=update_period
        )
