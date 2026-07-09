import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from petrecon.helpers.plotting import (
    plot_interactive, plot_static, create_sinogram,
    plot_scanner_geometry, plot_line_profile_with_gaussian_fit,
)
from shiny import App, ui, reactive, render

POSITIONS = [0, 32, 64, 96, 120]

# ── UI ───────────────────────────────────────────────────────────────────────
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_select("xy", "XY", choices={str(p): str(p) for p in POSITIONS}, selected="0"),
        ui.input_select("z",  "Z",  choices={str(p): str(p) for p in POSITIONS}, selected="0"),
        ui.input_select("filter_type", "LOR Filter",
                        choices={"none": "none", "midway": "midway"}, selected="none"),
        ui.hr(),
        ui.input_switch("smooth", "Smooth Recon", value=True),
        ui.input_slider("sigma", "Sigma", min=0.1, max=5.0, value=1.0, step=0.1),
        ui.input_slider("radius", "Radius", min=1, max=10, value=1, step=1),
    ),
    ui.tags.pre(ui.output_text("info")),
    ui.layout_columns(
        ui.output_plot("fig_gt", height="500px"),
        ui.output_plot("fig_pred", height="500px"),
        col_widths=[6, 6],
    ),
    ui.layout_columns(
        ui.output_plot("fig_profile_gt", height="400px"),
        ui.output_plot("fig_profile_pred", height="400px"),
        col_widths=[6, 6],
    ),
    title="PET Reconstruction Viewer",
)

# ── Server ───────────────────────────────────────────────────────────────────
def server(input, output, session):

    @reactive.calc
    def data():
        xy = int(input.xy())
        z  = int(input.z())
        source_pos = (xy, xy, z)

        if input.filter_type() == "none":
            path = "/home/h/Opengate/ragy_files/Point_Sources_Processed"
        else:
            path = "/home/h/Opengate/ragy_files/Point_Sources_r235_to_257mm"

        base = f"{path}/{source_pos[0]}_{source_pos[1]}_{source_pos[2]}"

        with open(f"{base}/gt/recon_config.pkl", "rb") as f:
            config = pickle.load(f)

        gt_image   = np.load(f"{base}/gt/reconstruction_image.npy")
        pred_image = np.load(f"{base}/pred/reconstruction_image.npy")

        if input.smooth():
            sigma = input.sigma()
            radius = int(input.radius())
            gt_image   = gaussian_filter(gt_image,   sigma=sigma, radius=radius)
            pred_image = gaussian_filter(pred_image, sigma=sigma, radius=radius)

        def hottest_coords(img):
            ix, iy, iz = np.unravel_index(np.argmax(img), img.shape)
            half = [s // 2 * config["VOXEL_SIZE_IN_MM"] for s in config["IMAGE_GRID_SIZE"]]
            return ix, iy, iz, ix - half[0], iy - half[1], iz - half[2]

        gt_ix, gt_iy, gt_iz, gt_xc, gt_yc, gt_zc     = hottest_coords(gt_image)
        pr_ix, pr_iy, pr_iz, pr_xc, pr_yc, pr_zc     = hottest_coords(pred_image)

        gt_dist   = np.sqrt((gt_xc - source_pos[0])**2 + (gt_yc - source_pos[1])**2 + (gt_zc - source_pos[2])**2)
        pred_dist = np.sqrt((pr_xc - source_pos[0])**2 + (pr_yc - source_pos[1])**2 + (pr_zc - source_pos[2])**2)

        return dict(
            source_pos=source_pos, config=config,
            gt_image=gt_image, pred_image=pred_image,
            gt_ix=gt_ix, gt_iy=gt_iy, gt_iz=gt_iz,
            gt_xc=gt_xc, gt_yc=gt_yc, gt_zc=gt_zc,
            pr_xc=pr_xc, pr_yc=pr_yc, pr_zc=pr_zc,
            gt_dist=gt_dist, pred_dist=pred_dist,
        )

    @render.text
    def info():
        d = data()
        gt_pass   = "✅" if d["gt_dist"]   < 3.0 else "❌"
        pred_pass = "✅" if d["pred_dist"] < 3.0 else "❌"
        return (
            f"Source position: {d['source_pos']}\n"
            f"Hottest GT voxel:   ({d['gt_xc']}, {d['gt_yc']}, {d['gt_zc']})  →  {d['gt_dist']:.2f}mm  {gt_pass}\n"
            f"Hottest PRED voxel: ({d['pr_xc']}, {d['pr_yc']}, {d['pr_zc']})  →  {d['pred_dist']:.2f}mm  {pred_pass}"
        )

    @render.plot
    def fig_gt():
        d = data()
        plot_static(d["gt_image"], title="gt_image", axis=2,
                    depth=d["gt_iz"], voxel_size=d["config"]["VOXEL_SIZE_IN_MM"])
        return plt.gcf()

    @render.plot
    def fig_pred():
        d = data()
        plot_static(d["pred_image"], title="prediction_image", axis=2,
                    depth=d["gt_iz"], voxel_size=d["config"]["VOXEL_SIZE_IN_MM"])
        return plt.gcf()

    @render.plot
    def fig_profile_gt():
        d = data()
        result = plot_line_profile_with_gaussian_fit(
            d["gt_image"], axis="z",
            coord1=d["gt_ix"], coord2=d["gt_iy"],
            voxel_size=d["config"]["VOXEL_SIZE_IN_MM"], show=False,
        )
        return result["fig"]

    @render.plot
    def fig_profile_pred():
        d = data()
        result = plot_line_profile_with_gaussian_fit(
            d["pred_image"], axis="z",
            coord1=d["gt_ix"], coord2=d["gt_iy"],
            voxel_size=d["config"]["VOXEL_SIZE_IN_MM"], show=False,
        )
        return result["fig"]


app = App(app_ui, server)