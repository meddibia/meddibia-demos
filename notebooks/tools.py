import matplotlib.pyplot as plt


def style_axis(
    ax: plt.axes,
    xlab: str = "",
    ylab: str = "",
    xgrid: bool = True,
    ygrid: bool = True,
    lab_size: int = 11,
) -> None:
    ax.minorticks_on()
    ax.spines[["top", "right"]].set_visible(False)

    if xlab != "":
        ax.set_xlabel(xlab, fontsize=lab_size)
    if ylab != "":
        ax.set_ylabel(ylab, fontsize=lab_size)

    if xgrid:
        ax.xaxis.grid(True, which="major", color="gray", alpha=0.6, linewidth=0.5)
        ax.xaxis.grid(
            True, which="minor", color="gray", alpha=0.6, linewidth=0.2, linestyle="--"
        )
    if ygrid:
        ax.yaxis.grid(True, which="major", color="gray", alpha=0.6, linewidth=0.5)
        ax.yaxis.grid(
            True, which="minor", color="gray", alpha=0.6, linewidth=0.2, linestyle="--"
        )
