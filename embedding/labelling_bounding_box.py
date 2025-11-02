import os
import cv2
import pandas as pd
import numpy as np
import traceback
import sys

class MultiRectGridLabelerSafe:
    def __init__(self, input_folder, output_csv="label_results.csv", grid_rows=8, grid_cols=8):
        self.input_folder = input_folder
        self.output_csv = output_csv
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        self.image_list = sorted([f for f in os.listdir(self.input_folder) if f.lower().endswith(exts)])
        if not self.image_list:
            raise SystemExit(f"No images found in {self.input_folder!r}")

        # per-image state
        self.boxes = []           # list of ((x1,y1),(x2,y2))
        self.dragging = False
        self.drag_start = None
        self.drag_current = None

        # index
        self.idx = 0

        # prepare dataframe (wide format)
        self.cell_cols = [f"cell_{i}" for i in range(self.grid_rows * self.grid_cols)]
        self.labels_df = self._load_or_create_labels()

        # window setup
        self.winname = "Multi-Rect Grid Labeler - draw boxes, Enter→save+next, d→del last, c→clear, s→save, q→quit"
        cv2.namedWindow(self.winname, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.winname, self._mouse_cb)

    def _load_or_create_labels(self):
        index = self.image_list
        if os.path.exists(self.output_csv):
            try:
                df_existing = pd.read_csv(self.output_csv)
            except Exception:
                print("Existing CSV found but couldn't read it. A new CSV will be created.")
                df_existing = None
            wide = pd.DataFrame(0, index=index, columns=self.cell_cols)
            if df_existing is not None:
                if "filename" in df_existing.columns:
                    for _, r in df_existing.iterrows():
                        fname = r.get("filename")
                        if fname in wide.index:
                            for c in self.cell_cols:
                                if c in r:
                                    try:
                                        wide.at[fname, c] = int(r[c])
                                    except Exception:
                                        wide.at[fname, c] = 0
            return wide
        else:
            return pd.DataFrame(0, index=index, columns=self.cell_cols)

    def _image_path(self, name):
        return os.path.join(self.input_folder, name)

    def _mouse_cb(self, event, x, y, flags, param):
        # safe mouse callback: only modifies instance attributes
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.drag_start = (int(x), int(y))
            self.drag_current = (int(x), int(y))
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.drag_current = (int(x), int(y))
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            end = (int(x), int(y))
            # canonicalize and store
            x1, y1 = self.drag_start
            x2, y2 = end
            self.boxes.append(((min(x1,x2), min(y1,y2)), (max(x1,x2), max(y1,y2))))
            self.drag_start = None
            self.drag_current = None

    def _compute_cell_size(self, img_w, img_h):
        cell_w = img_w // self.grid_cols
        cell_h = img_h // self.grid_rows
        return int(cell_w), int(cell_h)

    def _draw_grid(self, img):
        out = img.copy()
        h, w = out.shape[:2]
        cell_w, cell_h = self._compute_cell_size(w, h)
        col = (0, 255, 0)
        for r in range(1, self.grid_rows):
            y = r * cell_h
            cv2.line(out, (0, y), (w, y), col, 1)
        for c in range(1, self.grid_cols):
            x = c * cell_w
            cv2.line(out, (x, 0), (x, h), col, 1)
        return out

    def _overlay_rects_and_highlight(self, img):
        out = img.copy()
        h, w = out.shape[:2]
        cell_w, cell_h = self._compute_cell_size(w, h)

        # draw permanent boxes
        for (p1, p2) in self.boxes:
            cv2.rectangle(out, p1, p2, (0, 0, 255), 2)

        # draw dragging box
        if self.dragging and self.drag_start and self.drag_current:
            cv2.rectangle(out, self.drag_start, self.drag_current, (0, 0, 255), 1)

        # compute labels mask from boxes
        mask = np.zeros((self.grid_rows, self.grid_cols), dtype=np.uint8)
        for (p1, p2) in self.boxes:
            x_min, y_min = p1
            x_max, y_max = p2
            for r in range(self.grid_rows):
                for c in range(self.grid_cols):
                    gx1 = c * cell_w
                    gy1 = r * cell_h
                    gx2 = gx1 + cell_w
                    gy2 = gy1 + cell_h
                    if not (gx2 < x_min or gx1 > x_max or gy2 < y_min or gy1 > y_max):
                        mask[r, c] = 1

        # overlay shading
        if mask.sum() > 0:
            overlay = out.copy()
            alpha = 0.35
            for r in range(self.grid_rows):
                for c in range(self.grid_cols):
                    if mask[r, c] == 1:
                        x0 = c * cell_w
                        y0 = r * cell_h
                        x1 = x0 + cell_w
                        y1 = y0 + cell_h
                        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0,180,0), -1)
            out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
        return out

    def _boxes_to_flat_labels(self, img_w, img_h):
        cell_w, cell_h = self._compute_cell_size(img_w, img_h)
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=int)
        for (p1, p2) in self.boxes:
            x_min, y_min = p1
            x_max, y_max = p2
            for r in range(self.grid_rows):
                for c in range(self.grid_cols):
                    gx1 = c * cell_w
                    gy1 = r * cell_h
                    gx2 = gx1 + cell_w
                    gy2 = gy1 + cell_h
                    if not (gx2 < x_min or gx1 > x_max or gy2 < y_min or gy1 > y_max):
                        grid[r, c] = 1
        return grid.flatten().tolist()

    def save_csv(self):
        try:
            df_out = self.labels_df.reset_index().rename(columns={"index": "filename"})
            df_out = df_out[["filename"] + self.cell_cols]
            df_out.to_csv(self.output_csv, index=False)
            print(f"Saved {self.output_csv}")
        except Exception as e:
            print("Failed to save CSV:", e)

    def autosave_on_exception(self, e):
        # try to save partial progress
        try:
            bak = os.path.splitext(self.output_csv)[0] + "_backup.csv"
            df_out = self.labels_df.reset_index().rename(columns={"index": "filename"})
            df_out = df_out[["filename"] + self.cell_cols]
            df_out.to_csv(bak, index=False)
            print(f"Autosaved backup to {bak} due to exception.")
        except Exception as se:
            print("Autosave also failed:", se)
        print("Original exception:")
        traceback.print_exception(type(e), e, e.__traceback__, file=sys.stdout)

    def run(self):
        print("Instructions: draw boxes (click-drag). Multiple boxes allowed.")
        print("Enter -> save for this image and go to next (also autosaves CSV).")
        print("d -> delete last box, c -> clear all boxes, s -> save CSV, q/Esc -> quit.")
        try:
            while True:
                img_name = self.image_list[self.idx]
                path = self._image_path(img_name)
                img = cv2.imread(path)
                if img is None:
                    print("Could not read:", path)
                    if self.idx < len(self.image_list) - 1:
                        self.idx += 1
                        continue
                    else:
                        break

                while True:
                    disp = self._draw_grid(img)
                    disp = self._overlay_rects_and_highlight(disp)
                    status = f"{self.idx+1}/{len(self.image_list)}: {img_name}  (d=del last, c=clear, Enter=save+next, s=save, q=quit)"
                    cv2.putText(disp, status, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
                    cv2.imshow(self.winname, disp)

                    key = cv2.waitKey(20) & 0xFF
                    if key == 255:
                        continue

                    # quit
                    if key == ord('q') or key == 27:
                        print("Quit pressed.")
                        cv2.destroyAllWindows()
                        return

                    # delete last box
                    if key == ord('d'):
                        if self.boxes:
                            removed = self.boxes.pop()
                            print("Deleted last box:", removed)
                        else:
                            print("No boxes to delete.")
                        continue

                    # clear all
                    if key == ord('c'):
                        self.boxes = []
                        print("Cleared all boxes.")
                        continue

                    # save CSV now
                    if key == ord('s'):
                        self.save_csv()
                        continue

                    # ENTER: save labels for current image then go to next
                    if key in (13, 10):  # handle common Enter values
                        h, w = img.shape[:2]
                        flat = self._boxes_to_flat_labels(w, h)
                        # store row into df
                        self.labels_df.loc[img_name] = flat
                        # write CSV now
                        self.save_csv()
                        print(f"Saved labels for {img_name} -> {sum(flat)} marked.")
                        # reset boxes and advance
                        self.boxes = []
                        self.dragging = False
                        self.drag_start = None
                        self.drag_current = None
                        if self.idx < len(self.image_list) - 1:
                            self.idx += 1
                        else:
                            print("Last image processed. Exiting.")
                            cv2.destroyAllWindows()
                            return
                        break  # break inner loop to load next image

                    # next without saving
                    if key == ord('n'):
                        print("Next without saving.")
                        self.boxes = []
                        if self.idx < len(self.image_list) - 1:
                            self.idx += 1
                        else:
                            print("At last image.")
                        break

                    # previous
                    if key == ord('p'):
                        print("Previous")
                        self.boxes = []
                        if self.idx > 0:
                            self.idx -= 1
                        break

        except Exception as e:
            print("Exception occurred. Attempting to autosave progress...")
            self.autosave_on_exception(e)
            raise

if __name__ == "__main__":
    # ---- EDIT THIS PATH ----
    INPUT_FOLDER = r"C:\Users\ampat\Desktop\DS203\Project\train_temp"
    OUTPUT_CSV = "label_results.csv"
    # ------------------------
    lab = MultiRectGridLabelerSafe(INPUT_FOLDER, OUTPUT_CSV, grid_rows=8, grid_cols=8)
    lab.run()
