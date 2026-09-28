from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description="Local MRI and FreeSurfer viewer")
    parser.add_argument("--demo", action="store_true", help="Use a synthetic phantom, never patient data")
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--case", type=Path)
    parser.add_argument('--results',type=Path,nargs='+',help='Import local analysis Excel files after loading the patient')
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument('--smoke-results',action='store_true',help='Run synthetic result UI checks only')
    parser.add_argument('--smoke-dti',action='store_true',help='Run synthetic diffusion UI checks only')
    parser.add_argument('--smoke-views',action='store_true',help='Run synthetic ROI and panel checks only')
    parser.add_argument('--smoke-editing',action='store_true',help='Run synthetic tract editing and region deletion checks')
    parser.add_argument('--smoke-contacts',action='store_true',help='Run synthetic CT-section navigation and contact editing checks')
    parser.add_argument('--smoke-images',action='store_true',help='Run synthetic PET and bone CT import/display checks')
    parser.add_argument('--smoke-refinements',action='store_true',help='Run synthetic PET surface and contact review checks')
    parser.add_argument('--smoke-postop',action='store_true',help='Run synthetic postoperative MRI workflow checks')
    parser.add_argument('--smoke-layer-editing',action='store_true',help='Run synthetic image removal and postoperative mask editing checks')
    parser.add_argument("--capture-demo", type=Path, help="Save a synthetic screenshot during smoke testing")
    parser.add_argument("--offscreen", action="store_true")
    args = parser.parse_args()
    if args.smoke_results or args.smoke_dti or args.smoke_views or args.smoke_editing or args.smoke_contacts or args.smoke_images or args.smoke_refinements or args.smoke_postop or args.smoke_layer_editing:
        if not args.demo or args.case: parser.error('Focused smoke checks require --demo and no patient case.')
        args.smoke_test=True
    if args.capture_demo and (not args.demo or args.case):
        parser.error("Screenshot testing is restricted to the synthetic demo.")
    if args.offscreen:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication
    import vtk
    from .window import PROJECT_ROOT, ViewerWindow, log_error

    app = QApplication(sys.argv[:1])
    from . import APP_NAME
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("Local MRI Workspace")
    app.setFont(QFont("Yu Gothic UI", 10))
    reports = PROJECT_ROOT / "private_reports"
    reports.mkdir(exist_ok=True)
    vtk_log = vtk.vtkFileOutputWindow()
    vtk_log.SetFileName(str(reports / "vtk.log"))
    vtk.vtkOutputWindow.SetInstance(vtk_log)
    window = ViewerWindow(smoke=args.smoke_test, demo=args.demo)

    def exception_handler(kind, value, tb):
        import traceback
        with (reports / "viewer_error.log").open("a", encoding="utf-8") as stream:
            traceback.print_exception(kind, value, tb, file=stream)
        window.statusBar().showMessage("操作を完了できませんでした。ローカルのエラーログを確認してください。")
        if args.smoke_test:
            app.exit(2)
    sys.excepthook = exception_handler

    if args.smoke_test:
        from .smoke import run_ui_checks
        def test():
            try:
                if args.smoke_layer_editing:
                    from .layer_editing_smoke import run_layer_editing_checks
                    result=run_layer_editing_checks(window,args.capture_demo)
                elif args.smoke_postop:
                    from .postop_smoke import run_postop_checks
                    result=run_postop_checks(window,args.capture_demo)
                elif args.smoke_refinements:
                    from .review_refinements_smoke import run_review_refinement_checks
                    result=run_review_refinement_checks(window,args.capture_demo)
                elif args.smoke_images:
                    from .image_import_smoke import run_image_import_checks
                    result=run_image_import_checks(window,args.capture_demo)
                elif args.smoke_contacts:
                    from .contact_navigation_smoke import run_contact_navigation_checks
                    result=run_contact_navigation_checks(window,args.capture_demo)
                elif args.smoke_editing:
                    from .tract_editing_smoke import run_editing_checks
                    result=run_editing_checks(window,args.capture_demo)
                elif args.smoke_views:
                    from .view_workspace_smoke import run_view_checks
                    result=run_view_checks(window,args.capture_demo)
                elif args.smoke_dti:
                    from .diffusion_smoke import run_diffusion_checks
                    result=run_diffusion_checks(window,args.capture_demo)
                elif args.smoke_results:
                    from .results_smoke import run_result_checks
                    result=run_result_checks(window,args.capture_demo)
                else:
                    result = run_ui_checks(window, args.capture_demo)
                window.write_status("smoke_passed", result)
                print("UI_SMOKE_PASSED", flush=True)
                window.close()
                app.exit(0)
            except Exception:
                log_error()
                print("UI_SMOKE_FAILED; details stored locally", flush=True)
                window.close()
                app.exit(1)
        def start_test():
            window.scene_ready.disconnect(start_test)
            QTimer.singleShot(250, test)
        window.scene_ready.connect(start_test)
        window.scene_failed.connect(lambda _: app.exit(1))
        QTimer.singleShot(180000, lambda: app.exit(3))
    window.show()
    window.initialize()
    if args.results and not args.smoke_test:
        def import_results():
            window.scene_ready.disconnect(import_results)
            window.results.import_files(args.results)
        window.scene_ready.connect(import_results)
    if args.case:
        QTimer.singleShot(0, lambda: window.open_case(args.case))
    else:
        QTimer.singleShot(0, lambda: window.start_workspace(args.input_root or PROJECT_ROOT / "sample_images", args.demo))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
