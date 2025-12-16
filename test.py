import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def rotated_bounding_box(x, y, margin=0.0, plot=False, ax=None):
    """
    Lineer eğilimli (slope'lu) nokta dağılımı için
    doğru yönde döndürülmüş bounding box üretir.

    Parameters
    ----------
    x, y : array-like
        Nokta koordinatları (aynı uzunlukta)
    margin : float, optional
        Box etrafına eklenecek ek pay (hizalanmış uzayda)
    plot : bool, optional
        True ise örnek çizim yapar
    ax : matplotlib axis, optional
        Var olan axis üzerine çizmek için

    Returns
    -------
    rect : (4, 2) ndarray
        Döndürülmüş box köşe koordinatları
    slope : float
        Lineer fit eğimi
    intercept : float
        Lineer fit kesişimi
    """

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.size != y.size:
        raise ValueError("x ve y aynı uzunlukta olmalı")

    # -----------------------------
    # LINEER FIT
    # -----------------------------
    slope, intercept = np.polyfit(x, y, 1)
    theta = np.arctan(slope)

    # -----------------------------
    # ROTASYON MATRİSLERİ
    # -----------------------------
    R_align = np.array([
        [ np.cos(-theta), -np.sin(-theta)],
        [ np.sin(-theta),  np.cos(-theta)]
    ])

    R_back = np.array([
        [ np.cos(theta), -np.sin(theta)],
        [ np.sin(theta),  np.cos(theta)]
    ])

    pts = np.column_stack((x, y))

    # -----------------------------
    # NOKTALARI HİZALA
    # -----------------------------
    rot_pts = pts @ R_align.T

    xmin, ymin = rot_pts.min(axis=0) - margin
    xmax, ymax = rot_pts.max(axis=0) + margin

    # -----------------------------
    # HİZALANMIŞ BOX
    # -----------------------------
    rect_rot = np.array([
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax]
    ])

    # -----------------------------
    # GERİ DÖNDÜR
    # -----------------------------
    rect = rect_rot @ R_back.T

    # -----------------------------
    # OPSİYONEL ÇİZİM
    # -----------------------------
    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))

        ax.scatter(x, y, s=20, label="Points")

        poly = Polygon(rect, closed=True, fill=False,
                       edgecolor="green", linewidth=2,
                       label="Rotated box")
        ax.add_patch(poly)

        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, slope * xx + intercept, color="red",
                label="Linear fit")

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()

    return rect, slope, intercept

np.random.seed(0)

x = np.linspace(0, 10, 40)
y = 0.8 * x + 1.5 + np.random.normal(0, 0.25, size=len(x))

rect, slope, intercept = rotated_bounding_box(
    x, y,
    margin=0.2,
    plot=True
)

plt.show()
