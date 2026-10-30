"""机面上の画像座標と校正長方形のmm座標の変換。ロボット指令は扱わない。"""
import json
from pathlib import Path

import cv2
import numpy as np


def reference_points(width, height):
    if not np.isfinite([width, height]).all() or min(width, height) <= 0:
        raise ValueError('長方形の幅・高さは正の有限値にしてください。')
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=float)
    checks = np.array([[width/2, height/2], [width/2, 0], [width, height/2],
                       [width/2, height], [0, height/2]], dtype=float)
    return corners, checks


def fit_plane(corners_px, width, height):
    corners = np.asarray(corners_px, dtype=np.float32)
    world, _ = reference_points(width, height)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError('有限な四隅の座標が必要です。')
    if not cv2.isContourConvex(corners) or abs(cv2.contourArea(corners)) < 100:
        raise ValueError('四隅の順番・重複を確認してください。小さすぎる領域も使えません。')
    matrix = cv2.getPerspectiveTransform(corners, world.astype(np.float32))
    if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
        raise ValueError('座標変換を計算できません。四隅を確認してください。')
    return matrix


def project(matrix, points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('座標は有限なN×2配列にしてください。')
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ np.asarray(matrix).T
    if not np.isfinite(homogeneous).all() or np.any(np.abs(homogeneous[:, 2]) < 1e-10):
        raise ValueError('この点を座標変換できません。')
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def check_errors(matrix, check_px, width, height):
    _, expected = reference_points(width, height)
    if np.asarray(check_px).shape != (5, 2):
        raise ValueError('独立した確認点が5点必要です。')
    errors = np.linalg.norm(project(matrix, check_px) - expected, axis=1)
    return {'errors_mm': errors.tolist(), 'rmse_mm': float(np.sqrt(np.mean(errors**2))),
            'max_error_mm': float(errors.max())}


def load_calibration(path, camera, image_size):
    record = json.loads(Path(path).read_text())
    if record.get('schema_version') not in (1, 2) or record.get('coordinate_frame') != 'calibration_rectangle':
        raise ValueError('対応していない校正ファイルです。')
    if record.get('camera') != camera or record.get('image_size') != list(image_size):
        raise ValueError('校正時とカメラまたは解像度が異なります。再校正してください。')
    corners = record['corners_px']
    width, height = record['rectangle_mm']
    if record.get('status', 'passed') != 'passed':
        raise ValueError('合格した校正ファイルを指定してください。')
    if record['schema_version'] == 2:
        grid = record['grid']
        matrix, stats = fit_grid(record['grid_points_px'], grid['columns'], grid['rows'], grid['spacing_mm'])
        maximum = max(stats['fit']['max_error_mm'], stats['check']['max_error_mm'])
        expected_size = [(grid['columns']-1)*grid['spacing_mm'], (grid['rows']-1)*grid['spacing_mm']]
        ids = [0,grid['columns']-1,grid['rows']*grid['columns']-1,(grid['rows']-1)*grid['columns']]
        if not np.allclose([width,height], expected_size) or not np.allclose(corners,np.asarray(record['grid_points_px'])[ids]):
            raise ValueError('格子と校正領域の記録が一致しません。')
    else:
        matrix = fit_plane(corners, width, height)
        maximum = check_errors(matrix, record['check_points_px'], width, height)['max_error_mm']
    limit = record['max_allowed_error_mm']
    if not np.isfinite(limit) or not 0 < limit <= 5 or maximum > limit:
        raise ValueError('クリック点の誤差が許容値を超えています。再校正してください。')
    record['matrix'] = matrix
    return record


def plane_estimate(record, point):
    """校正した範囲内だけ変換する。物体の高さ補正は含まない。"""
    contour = np.asarray(record['corners_px'], dtype=np.float32)
    if cv2.pointPolygonTest(contour, tuple(map(float, point)), False) < 0:
        return None
    return project(record['matrix'], [point])[0].tolist()


def grid_reference(columns=8, rows=8, spacing_mm=30):
    if columns < 4 or rows < 4 or not np.isfinite(spacing_mm) or spacing_mm <= 0:
        raise ValueError('交点は各方向4点以上、間隔は正の有限値が必要です。')
    points = np.array([[c*spacing_mm, r*spacing_mm] for r in range(rows)
                       for c in range(columns)], dtype=float)
    # Interior odd/odd intersections are held out; all boundary points are fitted.
    checks = np.array([0 < r < rows-1 and 0 < c < columns-1 and r % 2 == 1 and c % 2 == 1
                       for r in range(rows) for c in range(columns)])
    return points, checks


def fit_grid(points_px, columns=8, rows=8, spacing_mm=30):
    expected, checks = grid_reference(columns, rows, spacing_mm)
    pixels = np.asarray(points_px, dtype=float)
    if pixels.shape != expected.shape or not np.isfinite(pixels).all():
        raise ValueError(f'有限な座標が{len(expected)}点必要です。')
    corners = pixels[[0, columns-1, rows*columns-1, (rows-1)*columns]]
    fit_plane(corners, (columns-1)*spacing_mm, (rows-1)*spacing_mm)
    matrix, _ = cv2.findHomography(pixels[~checks], expected[~checks], method=0)
    if matrix is None or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
        raise ValueError('格子から変換を求められません。クリック順を確認してください。')
    denominators = np.column_stack([corners, np.ones(4)]) @ matrix[2]
    if np.any(np.abs(denominators) < 1e-10) or not (np.all(denominators > 0) or np.all(denominators < 0)):
        raise ValueError('領域内で変換が発散します。クリック順を確認してください。')
    errors = np.linalg.norm(project(matrix, pixels)-expected, axis=1)
    def stats(values):
        return {'count':len(values), 'rmse_mm':float(np.sqrt(np.mean(values**2))),
                'max_error_mm':float(np.max(values))}
    return matrix, {'fit':stats(errors[~checks]), 'check':stats(errors[checks]),
                    'errors_mm':errors.tolist()}
