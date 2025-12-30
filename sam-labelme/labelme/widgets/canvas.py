import math

import imgviz
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
import numpy as np

import labelme.ai
import labelme.utils
from labelme import QT5, app
from labelme.logger import logger
from labelme.shape import Shape

# TODO(unknown):
# - [maybe] Find optimal epsilon value.


CURSOR_DEFAULT = QtCore.Qt.ArrowCursor
CURSOR_POINT = QtCore.Qt.PointingHandCursor
CURSOR_DRAW = QtCore.Qt.CrossCursor
CURSOR_MOVE = QtCore.Qt.ClosedHandCursor
CURSOR_GRAB = QtCore.Qt.OpenHandCursor

MOVE_SPEED = 5.0


class Canvas(QtWidgets.QWidget):
    zoomRequest = QtCore.Signal(int, QtCore.QPoint)
    scrollRequest = QtCore.Signal(int, int)
    newShape = QtCore.Signal()
    selectionChanged = QtCore.Signal(list)
    shapeMoved = QtCore.Signal()
    drawingPolygon = QtCore.Signal(bool)
    vertexSelected = QtCore.Signal(bool)
    mouseMoved = QtCore.Signal(QtCore.QPointF)
    segmentRemLabel = QtCore.Signal()
    segmentAddLabel = QtCore.Signal(object)
    initializeAiModelFlag = QtCore.Signal(object)

    CREATE, EDIT = 0, 1

    # polygon, rectangle, line, or point
    _createMode = "polygon"

    _fill_drawing = False

    def __init__(self, *args, **kwargs):
        self.epsilon = kwargs.pop("epsilon", 10.0)
        self.double_click = kwargs.pop("double_click", "close")
        if self.double_click not in [None, "close"]:
            raise ValueError(
                "Unexpected value for double_click event: {}".format(self.double_click)
            )
        self.num_backups = kwargs.pop("num_backups", 10)
        self._crosshair = kwargs.pop(
            "crosshair",
            {
                "polygon": False,
                "rectangle": True,
                "circle": False,
                "line": False,
                "point": False,
                "linestrip": False,
                "ai_polygon": False,
                "ai_bbox": True,
                "ai_rbox": True,
                "ai_mask": False,
            },
        )
        super(Canvas, self).__init__(*args, **kwargs)
        # Initialise local state.
        self.mode = self.EDIT
        self.shapes = []
        self.shapesBackups = []
        self.current = None
        self.selectedShapes = []  # save the selected shapes here
        self.selectedShapesCopy = []
        self.segmentShape = Shape()
        self.supplementShape = Shape(shape_type="points")
        self.canAddSupplementShapePoint = False
        # self.line represents:
        #   - createMode == 'polygon': edge from last point to current
        #   - createMode == 'rectangle': diagonal line of the rectangle
        #   - createMode == 'line': the line
        #   - createMode == 'point': the point
        self.line = Shape()
        self.prevPoint = QtCore.QPoint()
        self.prevMovePoint = QtCore.QPoint()
        self.offsets = QtCore.QPoint(), QtCore.QPoint()
        self.scale = 1.0
        self.pixmap = QtGui.QPixmap()
        self.visible = {}
        self._hideBackround = False
        self.hideBackround = False
        self.hShape = None
        self.prevhShape = None
        self.hVertex = None
        self.prevhVertex = None
        self.hEdge = None
        self.prevhEdge = None
        self.movingShape = False
        self.snapping = True
        self.hShapeIsSelected = False
        self._painter = QtGui.QPainter()
        self._cursor = CURSOR_DEFAULT
        self.forecastAiPolygon = False
        self.multiselected=False
        #self.multiselectedShapes = []
        self.multiRectangle = Shape(shape_type="rectangle")

        # Menus:
        # 0: right-click without selection and dragging of shapes
        # 1: right-click with selection and dragging of shapes
        self.menus = (QtWidgets.QMenu(), QtWidgets.QMenu())
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)

        self._ai_model = None

    def fillDrawing(self):
        return self._fill_drawing

    def setFillDrawing(self, value):
        self._fill_drawing = value

    @property
    def createMode(self):
        return self._createMode

    @createMode.setter
    def createMode(self, value):
        if value not in [
            "polygon",
            "rectangle",
            "circle",
            "line",
            "point",
            "linestrip",
            "ai_polygon",
            "ai_bbox",
            "ai_rbox",
            "ai_mask",
        ]:
            raise ValueError("Unsupported createMode: %s" % value)
        self._createMode = value

    def initializeAiModel(self, name):
        self.initializeAiModelFlag.emit("Start initializing the model: %s" % name)
        if name not in [model.name for model in labelme.ai.MODELS]:
            raise ValueError("Unsupported ai model: %s" % name)
        model = [model for model in labelme.ai.MODELS if model.name == name][0]

        if self._ai_model is not None and self._ai_model.name == model.name:
            logger.debug("AI model is already initialized: %r" % model.name)
        else:
            logger.debug("Initializing AI model: %r" % model.name)
            self._ai_model = model()

        if self.pixmap is None:
            logger.warning("Pixmap is not set yet")
            return

        self._ai_model.set_image(
            image=labelme.utils.img_qt_to_arr(self.pixmap.toImage())
        )
        self.setFocus()
        self.initializeAiModelFlag.emit("Finish initializing the model: %s" % name)

    def storeShapes(self):
        shapesBackup = []
        for shape in self.shapes:
            shapesBackup.append(shape.copy())
        if len(self.shapesBackups) > self.num_backups:
            self.shapesBackups = self.shapesBackups[-self.num_backups - 1:]
        self.shapesBackups.append(shapesBackup)

    @property
    def isShapeRestorable(self):
        # We save the state AFTER each edit (not before) so for an
        # edit to be undoable, we expect the CURRENT and the PREVIOUS state
        # to be in the undo stack.
        if len(self.shapesBackups) < 2:
            return False
        return True

    def restoreShape(self):
        # This does _part_ of the job of restoring shapes.
        # The complete process is also done in app.py::undoShapeEdit
        # and app.py::loadShapes and our own Canvas::loadShapes function.
        if not self.isShapeRestorable:
            return
        self.shapesBackups.pop()  # latest

        # The application will eventually call Canvas.loadShapes which will
        # push this right back onto the stack.
        shapesBackup = self.shapesBackups.pop()
        self.shapes = shapesBackup
        self.selectedShapes = []
        for shape in self.shapes:
            shape.selected = False
        self.update()

    def enterEvent(self, ev):
        self.setFocus()
        self.overrideCursor(self._cursor)

    def leaveEvent(self, ev):
        self.unHighlight()
        self.restoreCursor()

    def focusOutEvent(self, ev):
        self.restoreCursor()

    def isVisible(self, shape):
        return self.visible.get(shape, True)

    def drawing(self):
        return self.mode == self.CREATE

    def editing(self):
        return self.mode == self.EDIT

    def setEditing(self, value=True):
        self.mode = self.EDIT if value else self.CREATE
        if self.mode == self.EDIT:
            # CREATE -> EDIT
            self.repaint()  # clear crosshair
        else:
            # EDIT -> CREATE
            self.unHighlight()
            self.deSelectShape()

    def unHighlight(self):
        if self.hShape:
            self.hShape.highlightClear()
            self.update()
        self.prevhShape = self.hShape
        self.prevhVertex = self.hVertex
        self.prevhEdge = self.hEdge
        self.hShape = self.hVertex = self.hEdge = None

    def selectedVertex(self):
        return self.hVertex is not None

    def selectedEdge(self):
        return self.hEdge is not None

    def mouseMoveEvent(self, ev):
        """Update line with last point and current coordinates."""
        try:
            if QT5:
                pos = self.transformPos(ev.localPos())
            else:
                pos = self.transformPos(ev.posF())
        except AttributeError:
            return

        self.mouseMoved.emit(pos)

        self.prevMovePoint = pos
        self.restoreCursor()

        is_shift_pressed = ev.modifiers() & QtCore.Qt.ShiftModifier

        # Polygon drawing.
        if self.drawing():
            if self.createMode in ["ai_polygon", "ai_mask"]:
                self.line.shape_type = "points"
            elif self.createMode in ["ai_bbox"]:
                self.line.shape_type = "rectangle"
            elif self.createMode in ["ai_rbox"]:
                self.line.shape_type = "polygon"
            else:
                self.line.shape_type = self.createMode

            self.overrideCursor(CURSOR_DRAW)
            if not self.current:
                self.repaint()  # draw crosshair
                return

            if self.outOfPixmap(pos):
                # Don't allow the user to draw outside the pixmap.
                # Project the point to the pixmap's edges.
                pos = self.intersectionPoint(self.current[-1], pos)
            elif (
                    self.snapping
                    and len(self.current) > 1
                    and self.createMode == "polygon"
                    and self.closeEnough(pos, self.current[0])
            ):
                # Attract line to starting point and
                # colorise to alert the user.
                pos = self.current[0]
                self.overrideCursor(CURSOR_POINT)
                self.current.highlightVertex(0, Shape.NEAR_VERTEX)
            if self.createMode in ["polygon", "linestrip","ai_rbox"]:
                self.line.points = [self.current[-1], pos]
                self.line.point_labels = [1, 1]
            elif self.createMode in ["ai_polygon", "ai_mask"]:
                self.line.points = [self.current.points[-1], pos]
                self.line.point_labels = [
                    self.current.point_labels[-1],
                    0 if is_shift_pressed else 1,
                ]
            elif self.createMode == "ai_bbox":
                self.line.points = [self.current[0], pos]
                self.line.point_labels = [2, 3]
                self.line.close()
            elif self.createMode == "rectangle":
                self.line.points = [self.current[0], pos]
                self.line.point_labels = [1, 1]
                self.line.close()
            elif self.createMode == "circle":
                self.line.points = [self.current[0], pos]
                self.line.point_labels = [1, 1]
                self.line.shape_type = "circle"
            elif self.createMode == "line":
                self.line.points = [self.current[0], pos]
                self.line.point_labels = [1, 1]
                self.line.close()
            elif self.createMode == "point":
                self.line.points = [self.current[0]]
                self.line.point_labels = [1]
                self.line.close()
            assert len(self.line.points) == len(self.line.point_labels)
            self.repaint()
            self.current.highlightClear()
            return

        # Polygon copy moving.
        if QtCore.Qt.RightButton & ev.buttons() and not self.forecastAiPolygon:
            if self.selectedShapesCopy and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapesCopy, pos)
                self.repaint()
            elif self.selectedShapes:
                self.selectedShapesCopy = [s.copy() for s in self.selectedShapes]
                self.repaint()
            return

        # Polygon/Vertex moving.
        if QtCore.Qt.LeftButton & ev.buttons() and not self.forecastAiPolygon:
            if self.selectedVertex():
                self.boundedMoveVertex(pos)
                self.repaint()
                self.movingShape = True
            elif self.selectedShapes and self.prevPoint and self.selectedShapes[0].containsPoint(pos):
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapes, pos)
                self.repaint()
                self.movingShape = True
            return

        if self.multiselected and self.multiRectangle.points and ev.modifiers() & QtCore.Qt.ControlModifier:
            if len(self.multiRectangle.points) == 1:
                self.multiRectangle.addPoint(pos)
            else:
                assert len(self.multiRectangle.points) == 2
                self.multiRectangle.popPoint()
                self.multiRectangle.addPoint(pos)
            self.repaint()
            return

        # Just hovering over the canvas, 2 possibilities:
        # - Highlight shapes
        # - Highlight vertex
        # Update shape/vertex fill and tooltip value accordingly.
        self.setToolTip(self.tr("Image"))
        self.canAddSupplementShapePoint = True
        for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            index = shape.nearestVertex(pos, self.epsilon / self.scale)
            index_edge = shape.nearestEdge(pos, self.epsilon / self.scale)
            if index is not None:
                self.canAddSupplementShapePoint = False
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex = index
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge
                self.hEdge = None
                shape.highlightVertex(index, shape.MOVE_VERTEX)
                self.overrideCursor(CURSOR_POINT)
                self.setToolTip(self.tr("Click & drag to move point"))
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif index_edge is not None and shape.canAddPoint():
                self.canAddSupplementShapePoint = False
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge = index_edge
                self.overrideCursor(CURSOR_POINT)
                self.setToolTip(self.tr("Click to create point"))
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif shape.containsPoint(pos):
                self.canAddSupplementShapePoint = False
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge
                self.hEdge = None
                self.setToolTip(
                    self.tr("Click & drag to move shape '%s'") % shape.label
                )
                self.setStatusTip(self.toolTip())
                self.overrideCursor(CURSOR_GRAB)
                self.update()
                break
        else:  # Nothing found, clear highlights, reset state.
            self.unHighlight()
        self.vertexSelected.emit(self.hVertex is not None)

    def filter_shapes_inside_rect(self):
        assert len(self.multiRectangle.points)==2
        start_pos=self.multiRectangle.points[0]
        end_pos=self.multiRectangle.points[1]
        x1 = min(start_pos.x(), end_pos.x())
        y1 = min(start_pos.y(), end_pos.y())
        x2 = max(start_pos.x(), end_pos.x())
        y2 = max(start_pos.y(), end_pos.y())
        for shape in self.shapes:
            inside = True
            for point in shape.points:
                x = point.x()
                y = point.y()
                if not (x1 <= x <= x2 and y1 <= y <= y2):
                    inside = False
                    break
            if inside and shape not in self.selectedShapes:
                self.selectedShapes.append(shape)
        self.selectionChanged.emit(self.selectedShapes)

    def addPointToEdge(self):
        shape = self.prevhShape
        index = self.prevhEdge
        point = self.prevMovePoint
        if shape is None or index is None or point is None:
            return
        shape.insertPoint(index, point)
        shape.highlightVertex(index, shape.MOVE_VERTEX)
        self.hShape = shape
        self.hVertex = index
        self.hEdge = None
        self.movingShape = True

    def removeSelectedPoint(self):
        shape = self.prevhShape
        index = self.prevhVertex
        if shape is None or index is None:
            return
        shape.removePoint(index)
        shape.highlightClear()
        self.hShape = shape
        self.prevhVertex = None
        self.movingShape = True  # Save changes

    ## 切分的函数
    def segment(self, polygon, new_point1, new_point2):

        point1 = new_point1
        point2 = new_point2
        # print(new_point1, new_point2)

        # 找到原始多边形的边
        edges = []
        for i in range(len(polygon)):
            start = polygon[i]
            end = polygon[(i + 1) % len(polygon)]
            # print(start, end)
            edges.append((start, end))
        # print(edges)

        # 根据新增的点切分polygon
        polygon1 = []
        polygon2 = []
        flag_point = 0

        ## 判断点是否在边上
        def is_point_on_edge(point, edge_start, edge_end, tolerance=1.0):
            # 将点和边的端点转换为 NumPy 数组
            point = np.array(point)
            edge_start = np.array(edge_start)
            edge_end = np.array(edge_end)

            # 计算边的向量
            edge_vector = edge_end - edge_start
            point_vector = point - edge_start

            edge_vector = np.array([edge_vector.x(), edge_vector.y()])
            point_vector = np.array([point_vector.x(), point_vector.y()])
            # 检查 edge_vector 和 point_vector 是否是零维数组
            if edge_vector.ndim == 0:
                raise ValueError("edge_vector is a zero-dimensional array")
            if point_vector.ndim == 0:
                raise ValueError("point_vector is a zero-dimensional array")
            # 计算叉积
            cross_product = np.cross(edge_vector, point_vector)

            # 判断共线性
            if abs(cross_product) > tolerance:
                # print(abs(cross_product))
                return False  # 点不在边的延长线上

            # 判断点是否在边的范围内
            dot_product = np.dot(edge_vector, point_vector)
            if dot_product < 0:
                return False  # 点在边的起点之前
            if dot_product > np.dot(edge_vector, edge_vector):
                return False  # 点在边的终点之后

            return True  # 点在边上

        ## 点是边端点的处理
        def point_is_edge_end(point, edge_start, polygon1, polygon2, flag_point):
            # print(f"Point {point} is the end of the edge {(edge_start, point)}")
            # print(edge_start, edge_end)
            if flag_point == 0:
                if edge_start not in polygon1:
                    polygon1.append(edge_start)
                if point not in polygon1:
                    polygon1.append(point)
                if point not in polygon2:
                    polygon2.append(point)
            elif flag_point == 1:
                if edge_start not in polygon2:
                    polygon2.append(edge_start)
                if point not in polygon2:
                    polygon2.append(point)
                if point not in polygon1:
                    polygon1.append(point)

        ## 点在边上时的处理
        def point_is_on_edge(point, edge_start, edge_end, flag_point):
            # print(f"Point {point} is on the edge {(edge_start, edge_end)}")
            # print(edge_start, edge_end)
            # point = [round(point[0]), round(point[1])]
            if flag_point == 0:
                if edge_start not in polygon1:
                    polygon1.append(edge_start)
                if point not in polygon1:
                    polygon1.append(point)
                if point not in polygon2:
                    polygon2.append(point)
                if edge_end not in polygon2:
                    polygon2.append(edge_end)
            elif flag_point == 1:
                if edge_start not in polygon2:
                    polygon2.append(edge_start)
                if point not in polygon2:
                    polygon2.append(point)
                if point not in polygon1:
                    polygon1.append(point)
                if edge_end not in polygon1:
                    polygon1.append(edge_end)

        ## 点不在边上的处理
        def no_point_on_edge(edge_start, edge_end, flag_point):
            if flag_point == 0 or flag_point == 2:
                if edge_start not in polygon1:
                    polygon1.append(edge_start)
                if edge_end not in polygon1:
                    polygon1.append(edge_end)
            elif flag_point == 1:
                if edge_start not in polygon2:
                    polygon2.append(edge_start)
                if edge_end not in polygon2:
                    polygon2.append(edge_end)
            else:
                print("Error: Invalid flag value")

        ## 情况1：选择的两个点都是顶点
        if point1 in polygon and point2 in polygon:
            # print(f"Point {point1} and {point2} are in polygon")
            for point in polygon:
                if point == point1:
                    # print("point on edge end")
                    polygon1.append(point)
                    polygon2.append(point)
                    flag_point = flag_point + 1
                elif point == point2:
                    # print("point on edge end")
                    polygon1.append(point)
                    polygon2.append(point)
                    flag_point = flag_point + 1

                else:
                    if flag_point == 0 or flag_point == 2:
                        polygon1.append(point)
                    elif flag_point == 1:
                        polygon2.append(point)
                    else:
                        print("Error: Invalid flag value")

        ## 情况2：选择的point1是顶点
        elif point1 in polygon and point2 not in polygon:
            # print(f"Point1 {point1} is in polygon")
            for edge in edges:
                edge_start, edge_end = edge

                if edge_end == point1:
                    # print("point on edge end")
                    point_is_edge_end(point1, edge_start, polygon1, polygon2, flag_point)
                    flag_point = flag_point + 1
                elif is_point_on_edge(point2, edge_start, edge_end, tolerance=1e-6):
                    # print("point on edge")
                    point_is_on_edge(point2, edge_start, edge_end, flag_point)
                    flag_point = flag_point + 1
                else:
                    # print("no point on edge")
                    no_point_on_edge(edge_start, edge_end, flag_point)

        ## 情况3：选择的point2是顶点
        elif point1 not in polygon and point2 in polygon:
            # print(f"Point2 {point2} is in polygon")
            for edge in edges:
                edge_start, edge_end = edge

                if is_point_on_edge(point1, edge_start, edge_end, tolerance=1e-6):
                    # print("point on edge")
                    point_is_on_edge(point1, edge_start, edge_end, flag_point)
                    flag_point = flag_point + 1
                elif edge_end == point2:
                    # print("point on edge end")
                    point_is_edge_end(point2, edge_start, polygon1, polygon2, flag_point)
                    flag_point = flag_point + 1
                else:
                    # print("no point on edge")
                    no_point_on_edge(edge_start, edge_end, flag_point)

        ## 情况4：选择的点都在边上
        elif point1 not in polygon and point2 not in polygon:
            for edge in edges:
                edge_start, edge_end = edge

                if is_point_on_edge(point1, edge_start, edge_end, tolerance=1e-6):
                    point_is_on_edge(point1, edge_start, edge_end, flag_point)
                    flag_point = flag_point + 1
                elif is_point_on_edge(point2, edge_start, edge_end, tolerance=1e-6):
                    point_is_on_edge(point2, edge_start, edge_end, flag_point)
                    flag_point = flag_point + 1
                else:
                    no_point_on_edge(edge_start, edge_end, flag_point)

        return polygon1, polygon2

    def mousePressEvent(self, ev):
        if QT5:
            pos = self.transformPos(ev.localPos())
        else:
            pos = self.transformPos(ev.posF())

        is_shift_pressed = ev.modifiers() & QtCore.Qt.ShiftModifier

        if ev.button() == QtCore.Qt.LeftButton:
            if self.drawing():
                if self.current:
                    # Add point to existing shape.
                    if self.createMode == "polygon":
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if self.current.isClosed():
                            self.finalise()
                    elif self.createMode in ["rectangle", "circle", "line","ai_bbox"]:
                        assert len(self.current.points) == 1
                        self.current.points = self.line.points
                        self.current.point_labels = self.line.point_labels
                        self.finalise()
                    elif self.createMode == "linestrip":
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if int(ev.modifiers()) == QtCore.Qt.ControlModifier:
                            self.finalise()
                    elif self.createMode == "ai_rbox":
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if len(self.current.points) == 4:
                            self.finalise()
                    elif self.createMode in ["ai_polygon", "ai_mask"] and not self.forecastAiPolygon:
                        self.current.addPoint(
                            self.line.points[1],
                            label=self.line.point_labels[1],
                        )
                        self.line.points[0] = self.current.points[-1]
                        self.line.point_labels[0] = self.current.point_labels[-1]
                        if ev.modifiers() & QtCore.Qt.ControlModifier:
                            self.finalise()
                elif not self.outOfPixmap(pos):
                    # Create new shape.

                    if self.createMode in ["ai_polygon", "ai_mask"]:
                        shape_type = "points"
                    elif self.createMode in ["ai_bbox"]:
                        shape_type = "rectangle"
                    elif self.createMode in ["ai_rbox"]:
                        shape_type = "polygon"
                    else:
                        shape_type = self.createMode
                    self.current = Shape(
                        shape_type=shape_type
                    )

                    self.current.addPoint(pos, label=0 if is_shift_pressed else 1)
                    if self.createMode == "point":
                        self.finalise()
                    elif (
                            self.createMode in ["ai_polygon", "ai_mask"]
                            and ev.modifiers() & QtCore.Qt.ControlModifier
                    ):
                        self.finalise()
                    else:
                        if self.createMode == "circle":
                            self.current.shape_type = "circle"
                        self.line.points = [pos, pos]
                        if (
                                self.createMode in ["ai_polygon", "ai_mask"]
                                and is_shift_pressed
                        ):
                            self.line.point_labels = [0, 0]
                        else:
                            self.line.point_labels = [1, 1]
                        self.setHiding()
                        self.drawingPolygon.emit(True)
                        self.update()
            elif self.editing() and not self.forecastAiPolygon:
                if self.selectedEdge():#左键点击+在边附近
                    if ev.modifiers() & QtCore.Qt.ControlModifier:#按住ctrl--加shape分割点
                        self.segmentShape.points.append(self.prevMovePoint)
                        self.performSegment()
                    else:#加点到边
                        self.addPointToEdge()
                elif self.selectedVertex():#左键点击+在点附近
                    if int(ev.modifiers()) == QtCore.Qt.ShiftModifier:#按住shift--删点
                        # Delete point if: left-click + SHIFT on a point
                        self.removeSelectedPoint()
                    elif int(ev.modifiers()) == QtCore.Qt.ControlModifier:#按住ctrl--加shape分割点
                        self.segmentShape.points.append(self.hShape.points[self.hVertex])
                        self.performSegment()
                elif self.multiselected:
                    #if not self.multiselected:
                        #self.selectionChanged.emit([])
                        #for shape in self.shapes:
                            #shape.selected=False
                    #self.multiselected=True
                    rec_flag=True
                    if len(self.multiRectangle.points)<=0:
                        for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
                            # Look for a nearby vertex to highlight. If that fails,
                            # check if we happen to be inside a shape.
                            index = shape.nearestVertex(pos, self.epsilon / self.scale)
                            index_edge = shape.nearestEdge(pos, self.epsilon / self.scale)
                            if index is not None or index_edge is not None or shape.containsPoint(pos):
                                if shape in self.selectedShapes:
                                    #self.selectedShapes.remove(shape)
                                    newshape =self.selectedShapes.copy()
                                    newshape.remove(shape)
                                    self.selectionChanged.emit(newshape)
                                    #self.multiselectedShapes.remove(shape)
                                    #shape.selected = False
                                else:
                                    self.selectionChanged.emit(self.selectedShapes+[shape])
                                    #self.multiselectedShapes.append(shape)
                                    #shape.selected = True
                                self.update()
                                rec_flag = False
                                break
                    if rec_flag:
                        if len(self.multiRectangle.points)==0:
                            self.multiRectangle.addPoint(pos)
                        if len(self.multiRectangle.points)==2:
                            self.filter_shapes_inside_rect()
                            self.multiRectangle.points.clear()
                            self.multiRectangle.point_labels.clear()


                group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier

                if not ev.modifiers() & QtCore.Qt.ControlModifier:
                    prevSelectedShapes = self.selectedShapes.copy()
                    self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                    self.prevPoint = pos
                    if prevSelectedShapes and prevSelectedShapes == self.selectedShapes and not self.forecastAiPolygon:
                        if self.canAddSupplementShapePoint:
                            self.supplementShape.addPoint(
                                pos,
                                label=0 if is_shift_pressed else 1,
                            )

                self.repaint()
        elif ev.button() == QtCore.Qt.RightButton and self.editing() and not self.forecastAiPolygon:
            group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier
            if not self.selectedShapes or (
                    self.hShape is not None and self.hShape not in self.selectedShapes
            ):
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self.repaint()
            self.prevPoint = pos

    def mouseReleaseEvent(self, ev):
        if QT5:
            pos = self.transformPos(ev.localPos())
        else:
            pos = self.transformPos(ev.posF())

        if not self.forecastAiPolygon:
            if ev.button() == QtCore.Qt.RightButton:
                menu = self.menus[len(self.selectedShapesCopy) > 0]
                self.restoreCursor()
                if not menu.exec_(self.mapToGlobal(ev.pos())) and self.selectedShapesCopy:
                    # Cancel the move by deleting the shadow copy.
                    self.selectedShapesCopy = []
                    self.repaint()
            elif ev.button() == QtCore.Qt.LeftButton:
                if self.editing():
                    if (
                            self.hShape is not None
                            and self.hShapeIsSelected
                            and not self.movingShape
                    ):
                        self.selectionChanged.emit(
                            [x for x in self.selectedShapes if x != self.hShape]
                        )
                        self.repaint()




            if self.movingShape and self.hShape:
                index = self.shapes.index(self.hShape)
                if self.shapesBackups[-1][index].points != self.shapes[index].points:
                    self.storeShapes()
                    self.shapeMoved.emit()

                self.movingShape = False

    def endMove(self, copy):
        assert self.selectedShapes and self.selectedShapesCopy
        assert len(self.selectedShapesCopy) == len(self.selectedShapes)
        if copy:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.shapes.append(shape)
                self.selectedShapes[i].selected = False
                self.selectedShapes[i] = shape
        else:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.selectedShapes[i].points = shape.points
        self.selectedShapesCopy = []
        self.repaint()
        self.storeShapes()
        return True

    def hideBackroundShapes(self, value):
        self.hideBackround = value
        if self.selectedShapes:
            # Only hide other shapes if there is a current selection.
            # Otherwise the user will not be able to select a shape.
            self.setHiding(True)
            self.update()

    def setHiding(self, enable=True):
        self._hideBackround = self.hideBackround if enable else False

    def canCloseShape(self):
        return self.drawing() and (
                (self.current and len(self.current) > 2)
                or
                (self.current and self.createMode in ["ai_polygon", "ai_bbox","ai_rbox", "ai_mask"])
        )

    def mouseDoubleClickEvent(self, ev):
        if self.double_click != "close":
            return

        if (
                self.createMode == "polygon" and self.canCloseShape()
        ) or self.createMode in ["ai_polygon", "ai_mask"]:
            self.finalise()

    def selectShapes(self, shapes):
        self.setHiding()
        self.selectionChanged.emit(shapes)
        self.update()

    def selectShapePoint(self, point, multiple_selection_mode):
        """Select the first shape created which contains this point."""
        if self.selectedVertex():  # A vertex is marked for selection.
            index, shape = self.hVertex, self.hShape
            shape.highlightVertex(index, shape.MOVE_VERTEX)
        else:
            for shape in reversed(self.shapes):
                if self.isVisible(shape) and shape.containsPoint(point):
                    self.setHiding()
                    if shape not in self.selectedShapes:
                        self.supplementShape.points.clear()
                        self.supplementShape.point_labels.clear()
                        if multiple_selection_mode:
                            self.selectionChanged.emit(self.selectedShapes + [shape])
                        else:
                            self.selectionChanged.emit([shape])
                        self.hShapeIsSelected = False
                    else:
                        self.supplementShape.points.clear()
                        self.supplementShape.point_labels.clear()
                        self.selectionChanged.emit([shape])
                        self.hShapeIsSelected = True
                    self.calculateOffsets(point)
                    return

        # self.deSelectShape()

    def calculateOffsets(self, point):
        left = self.pixmap.width() - 1
        right = 0
        top = self.pixmap.height() - 1
        bottom = 0
        for s in self.selectedShapes:
            rect = s.boundingRect()
            if rect.left() < left:
                left = rect.left()
            if rect.right() > right:
                right = rect.right()
            if rect.top() < top:
                top = rect.top()
            if rect.bottom() > bottom:
                bottom = rect.bottom()

        x1 = left - point.x()
        y1 = top - point.y()
        x2 = right - point.x()
        y2 = bottom - point.y()
        self.offsets = QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)

    def boundedMoveVertex(self, pos):
        index, shape = self.hVertex, self.hShape
        point = shape[index]
        if self.outOfPixmap(pos):
            pos = self.intersectionPoint(point, pos)
        shape.moveVertexBy(index, pos - point)

    def boundedMoveShapes(self, shapes, pos):
        if self.outOfPixmap(pos):
            return False  # No need to move
        o1 = pos + self.offsets[0]
        if self.outOfPixmap(o1):
            pos -= QtCore.QPointF(min(0, o1.x()), min(0, o1.y()))
        o2 = pos + self.offsets[1]
        if self.outOfPixmap(o2):
            pos += QtCore.QPointF(
                min(0, self.pixmap.width() - o2.x()),
                min(0, self.pixmap.height() - o2.y()),
            )
        # XXX: The next line tracks the new position of the cursor
        # relative to the shape, but also results in making it
        # a bit "shaky" when nearing the border and allows it to
        # go outside of the shape's area for some reason.
        # self.calculateOffsets(self.selectedShapes, pos)
        dp = pos - self.prevPoint
        if dp:
            for shape in shapes:
                shape.moveBy(dp)
            if self.segmentShape.points:
                self.segmentShape.moveBy(dp)
            self.prevPoint = pos
            return True
        return False

    def deSelectShape(self):
        if self.selectedShapes:
            self.setHiding(False)
            self.selectionChanged.emit([])
            self.hShapeIsSelected = False
            self.supplementShape.points.clear()
            self.supplementShape.point_labels.clear()
            self.update()

    def deleteSelected(self):
        deleted_shapes = []
        """
                if self.multiselected:
            for shape in self.multiselectedShapes:
                self.shapes.remove(shape)
                deleted_shapes.append(shape)
            self.storeShapes()
            self.multiselectedShapes = []
            self.update()
        el
        """
        if self.selectedShapes:
            for shape in self.selectedShapes:
                self.shapes.remove(shape)
                deleted_shapes.append(shape)
            #self.storeShapes()
            self.selectedShapes = []
            self.update()
        return deleted_shapes

    def deleteShape(self, shape):
        if shape in self.selectedShapes:
            self.selectedShapes.remove(shape)
        if shape in self.shapes:
            self.shapes.remove(shape)
        self.storeShapes()
        self.update()

    def duplicateSelectedShapes(self):
        if self.selectedShapes:
            self.selectedShapesCopy = [s.copy() for s in self.selectedShapes]
            self.boundedShiftShapes(self.selectedShapesCopy)
            self.endMove(copy=True)
        return self.selectedShapes

    def boundedShiftShapes(self, shapes):
        # Try to move in one direction, and if it fails in another.
        # Give up if both fail.
        point = shapes[0][0]
        offset = QtCore.QPointF(2.0, 2.0)
        self.offsets = QtCore.QPoint(), QtCore.QPoint()
        self.prevPoint = point
        if not self.boundedMoveShapes(shapes, point - offset):
            self.boundedMoveShapes(shapes, point + offset)

    def paintEvent(self, event):
        if not self.pixmap:
            return super(Canvas, self).paintEvent(event)

        p = self._painter
        p.begin(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        p.scale(self.scale, self.scale)
        p.translate(self.offsetToCenter())

        p.drawPixmap(0, 0, self.pixmap)

        # draw crosshair
        if (
                self._crosshair[self._createMode]
                and self.drawing()
                and self.prevMovePoint
                and not self.outOfPixmap(self.prevMovePoint)
        ):
            p.setPen(QtGui.QColor(0, 0, 0))
            p.drawLine(
                0,
                int(self.prevMovePoint.y()),
                self.width() - 1,
                int(self.prevMovePoint.y()),
            )
            p.drawLine(
                int(self.prevMovePoint.x()),
                0,
                int(self.prevMovePoint.x()),
                self.height() - 1,
            )

        #if self.multiselectedShapes:
            #for shape in self.multiselectedShapes:
                #shape.selected=True


        Shape.scale = self.scale
        for shape in self.shapes:
            if (shape.selected or not self._hideBackround) and self.isVisible(shape):
                shape.fill = self.fillDrawing() or shape.selected or shape == self.hShape
                shape.paint(p)
        if self.current:
            self.current.paint(p)
            assert len(self.line.points) == len(self.line.point_labels)
            self.line.paint(p)
        if self.selectedShapesCopy:
            for s in self.selectedShapesCopy:
                s.paint(p)
        if self.segmentShape.points:
            self.segmentShape.paint(p)
        if self.supplementShape.points:
            self.supplementShape.paint(p)
        if self.multiRectangle.points:
            self.multiRectangle.paint(p)


        if (
                self.fillDrawing()
                and self.createMode == "polygon"
                and self.current is not None
                and len(self.current.points) >= 2
        ):
            drawing_shape = self.current.copy()
            if drawing_shape.fill_color.getRgb()[3] == 0:
                logger.warning(
                    "fill_drawing=true, but fill_color is transparent,"
                    " so forcing to be opaque."
                )
                drawing_shape.fill_color.setAlpha(64)
            drawing_shape.addPoint(self.line[1])
            drawing_shape.fill = True
            drawing_shape.paint(p)
        elif ((self.createMode == "ai_polygon" and self.current is not None) or (
                self.editing() and self.selectedShapes and self.supplementShape.points)) and self.forecastAiPolygon:
            if self.drawing():
                drawing_shape = self.current.copy()
            elif self.editing():
                drawing_shape = self.selectedShapes[0].copy()
                drawing_shape.points.extend(self.supplementShape.points)
                drawing_shape.point_labels.extend(self.supplementShape.point_labels)
            """
            drawing_shape.addPoint(
                point=self.line.points[1],
                label=self.line.point_labels[1],
            )
            """
            if self._ai_model is None:
                self.initializeAiModel(name="SAM_B")
            bool_mask, points = self._ai_model.predict_polygon_from_points(
                points=[[point.x(), point.y()] for point in drawing_shape.points],
                point_labels=drawing_shape.point_labels,
            )
            if points is not None and len(points) > 2:
                drawing_shape.setShapeRefined(
                    shape_type="polygon",
                    points=[QtCore.QPointF(point[0], point[1]) for point in points],
                    point_labels=[1] * len(points),
                )
                drawing_shape.fill = self.fillDrawing()
                drawing_shape.selected = True
                drawing_shape.paint(p)
        elif self.createMode == "ai_mask" and self.current is not None:
            drawing_shape = self.current.copy()
            drawing_shape.addPoint(
                point=self.line.points[1],
                label=self.line.point_labels[1],
            )
            mask = self._ai_model.predict_mask_from_points(
                points=[[point.x(), point.y()] for point in drawing_shape.points],
                point_labels=drawing_shape.point_labels,
            )
            y1, x1, y2, x2 = imgviz.instances.masks_to_bboxes([mask])[0].astype(int)
            drawing_shape.setShapeRefined(
                shape_type="mask",
                points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
                point_labels=[1, 1],
                mask=mask[y1: y2 + 1, x1: x2 + 1],
            )
            drawing_shape.selected = True
            drawing_shape.paint(p)

        p.end()

    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = super(Canvas, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QtCore.QPointF(x, y)

    def outOfPixmap(self, p):
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w - 1 and 0 <= p.y() <= h - 1)

    def finalise(self):
        assert self.current
        if self.createMode == "ai_polygon":
            # convert points to polygon by an AI model
            assert self.current.shape_type == "points"
            self.forecastAiPolygon = False
            input_points = self.current.points.copy()
            bool_mask, points = self._ai_model.predict_polygon_from_points(
                points=[[point.x(), point.y()] for point in self.current.points],
                point_labels=self.current.point_labels,
            )
            self.current.setShapeRefined(
                points=[QtCore.QPointF(point[0], point[1]) for point in points],
                point_labels=[1] * len(points),
                shape_type="polygon",
                input_points=input_points,
                output_mask=bool_mask
            )
        elif self.createMode == "ai_bbox":
            # convert points to polygon by an AI model
            assert self.current.shape_type == "rectangle"
            self.forecastAiPolygon = False
            input_points = self.current.points.copy()
            bool_mask, points = self._ai_model.predict_polygon_from_points(
                points=[[point.x(), point.y()] for point in self.current.points],#[[x,y],[x,y]......]
                point_labels=[2,3],
            )
            self.current.setShapeRefined(
                points=[QtCore.QPointF(point[0], point[1]) for point in points],
                point_labels=[1] * len(points),
                shape_type="polygon",
                input_points=input_points,
                output_mask=bool_mask
            )
        elif self.createMode == "ai_rbox":
            # convert points to polygon by an AI model
            assert self.current.shape_type == "polygon"
            self.forecastAiPolygon = False
            input_points = self.current.points.copy()
            bool_mask, points = self._ai_model.predict_polygon_from_points(
                points=[[point.x(), point.y()] for point in self.current.points],#[[x,y],[x,y]......]
                point_labels=[4,5,6,7],
            )
            self.current.setShapeRefined(
                points=[QtCore.QPointF(point[0], point[1]) for point in points],
                point_labels=[1] * len(points),
                shape_type="polygon",
                input_points=input_points,
                output_mask=bool_mask
            )
        elif self.createMode == "ai_mask":
            # convert points to mask by an AI model
            assert self.current.shape_type == "points"
            mask = self._ai_model.predict_mask_from_points(
                points=[[point.x(), point.y()] for point in self.current.points],
                point_labels=self.current.point_labels,
            )
            y1, x1, y2, x2 = imgviz.instances.masks_to_bboxes([mask])[0].astype(int)
            self.current.setShapeRefined(
                shape_type="mask",
                points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
                point_labels=[1, 1],
                mask=mask[y1: y2 + 1, x1: x2 + 1],
            )
        self.current.close()

        self.shapes.append(self.current)
        self.storeShapes()
        self.current = None
        self.setHiding(False)
        self.newShape.emit()
        self.update()

    def closeEnough(self, p1, p2):
        # d = distance(p1 - p2)
        # m = (p1-p2).manhattanLength()
        # print "d %.2f, m %d, %.2f" % (d, m, d - m)
        # divide by scale to allow more precision when zoomed in
        return labelme.utils.distance(p1 - p2) < (self.epsilon / self.scale)

    def intersectionPoint(self, p1, p2):
        # Cycle through each image edge in clockwise fashion,
        # and find the one intersecting the current line segment.
        # http://paulbourke.net/geometry/lineline2d/
        size = self.pixmap.size()
        points = [
            (0, 0),
            (size.width() - 1, 0),
            (size.width() - 1, size.height() - 1),
            (0, size.height() - 1),
        ]
        # x1, y1 should be in the pixmap, x2, y2 should be out of the pixmap
        x1 = min(max(p1.x(), 0), size.width() - 1)
        y1 = min(max(p1.y(), 0), size.height() - 1)
        x2, y2 = p2.x(), p2.y()
        d, i, (x, y) = min(self.intersectingEdges((x1, y1), (x2, y2), points))
        x3, y3 = points[i]
        x4, y4 = points[(i + 1) % 4]
        if (x, y) == (x1, y1):
            # Handle cases where previous point is on one of the edges.
            if x3 == x4:
                return QtCore.QPointF(x3, min(max(0, y2), max(y3, y4)))
            else:  # y3 == y4
                return QtCore.QPointF(min(max(0, x2), max(x3, x4)), y3)
        return QtCore.QPointF(x, y)

    def intersectingEdges(self, point1, point2, points):
        """Find intersecting edges.

        For each edge formed by `points', yield the intersection
        with the line segment `(x1,y1) - (x2,y2)`, if it exists.
        Also return the distance of `(x2,y2)' to the middle of the
        edge along with its index, so that the one closest can be chosen.
        """
        (x1, y1) = point1
        (x2, y2) = point2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            if denom == 0:
                # This covers two cases:
                #   nua == nub == 0: Coincident
                #   otherwise: Parallel
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QtCore.QPointF((x3 + x4) / 2, (y3 + y4) / 2)
                d = labelme.utils.distance(m - QtCore.QPointF(x2, y2))
                yield d, i, (x, y)

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(Canvas, self).minimumSizeHint()

    def wheelEvent(self, ev):
        if QT5:
            mods = ev.modifiers()
            delta = ev.angleDelta()
            if QtCore.Qt.ControlModifier == int(mods):
                # with Ctrl/Command key
                # zoom
                self.zoomRequest.emit(delta.y(), ev.pos())
            else:
                # scroll
                self.scrollRequest.emit(delta.x(), QtCore.Qt.Horizontal)
                self.scrollRequest.emit(delta.y(), QtCore.Qt.Vertical)
        else:
            if ev.orientation() == QtCore.Qt.Vertical:
                mods = ev.modifiers()
                if QtCore.Qt.ControlModifier == int(mods):
                    # with Ctrl/Command key
                    self.zoomRequest.emit(ev.delta(), ev.pos())
                else:
                    self.scrollRequest.emit(
                        ev.delta(),
                        QtCore.Qt.Horizontal
                        if (QtCore.Qt.ShiftModifier == int(mods))
                        else QtCore.Qt.Vertical,
                    )
            else:
                self.scrollRequest.emit(ev.delta(), QtCore.Qt.Horizontal)
        ev.accept()

    def moveByKeyboard(self, offset):
        if self.selectedShapes and not self.forecastAiPolygon:
            self.boundedMoveShapes(self.selectedShapes, self.prevPoint + offset)
            self.repaint()
            self.movingShape = True

    def keyPressEvent(self, ev):
        modifiers = ev.modifiers()
        key = ev.key()
        if self.drawing():
            if key == QtCore.Qt.Key_Escape and self.current:
                self.current = None
                self.drawingPolygon.emit(False)
                self.update()
            elif key == QtCore.Qt.Key_Return and self.canCloseShape():
                self.finalise()
            elif modifiers == QtCore.Qt.AltModifier:
                self.snapping = False
            elif key == QtCore.Qt.Key_Space and self.createMode == "ai_polygon" and not ev.isAutoRepeat():
                self.forecastAiPolygon = True
                self.repaint()
        elif self.editing():
            if key == QtCore.Qt.Key_Up:
                self.moveByKeyboard(QtCore.QPointF(0.0, -MOVE_SPEED))
            elif key == QtCore.Qt.Key_Down:
                self.moveByKeyboard(QtCore.QPointF(0.0, MOVE_SPEED))
            elif key == QtCore.Qt.Key_Left:
                self.moveByKeyboard(QtCore.QPointF(-MOVE_SPEED, 0.0))
            elif key == QtCore.Qt.Key_Right:
                self.moveByKeyboard(QtCore.QPointF(MOVE_SPEED, 0.0))
            elif key == QtCore.Qt.Key_Space and not ev.isAutoRepeat():
                self.forecastAiPolygon = True
                self.repaint()
            elif key == QtCore.Qt.Key_Return and self.supplementShape.points:
                self.getNewPolygonFromPolygonAndPoints()
                self.repaint()
            elif key == QtCore.Qt.Key_Escape and self.segmentShape.points:
                self.segmentShape.points.pop()
                self.repaint()
            elif key == QtCore.Qt.Key_Escape and self.supplementShape.points and not self.forecastAiPolygon:
                self.supplementShape.points.pop()
                self.supplementShape.point_labels.pop()
                self.repaint()
            elif key == QtCore.Qt.Key_Escape and self.selectedShapes:
                self.multiselected=False
                self.multiRectangle.points.clear()
                self.multiRectangle.point_labels.clear()
                #for shape in self.multiselectedShapes:
                    #shape.selected=False
                #self.multiselectedShapes.clear()
                self.selectionChanged.emit([])
                self.repaint()
            elif key == QtCore.Qt.Key_Control:
                self.multiselected = True
    def keyReleaseEvent(self, ev):
        modifiers = ev.modifiers()
        if self.drawing():
            if ev.key() == QtCore.Qt.Key_Space and self.createMode == "ai_polygon" and not ev.isAutoRepeat():
                self.forecastAiPolygon = False
                self.repaint()
            if int(modifiers) == 0:
                self.snapping = True
        elif self.editing():
            if ev.key() == QtCore.Qt.Key_Space and not ev.isAutoRepeat():
                self.forecastAiPolygon = False
                self.repaint()
            if self.movingShape and self.selectedShapes:
                index = self.shapes.index(self.selectedShapes[0])
                if self.shapesBackups[-1][index].points != self.shapes[index].points:
                    self.storeShapes()
                    self.shapeMoved.emit()
                self.movingShape = False
            if ev.key() == QtCore.Qt.Key_Control  and self.multiselected:
                self.multiselected = False
                self.multiRectangle.points.clear()
                self.multiRectangle.point_labels.clear()
                self.repaint()

    def setLastLabel(self, text, flags):
        assert text
        self.shapes[-1].label = text
        self.shapes[-1].flags = flags
        self.shapesBackups.pop()
        self.storeShapes()
        return self.shapes[-1]

    def undoLastLine(self):
        assert self.shapes
        self.current = self.shapes.pop()
        self.current.setOpen()
        self.current.restoreShapeRaw()
        if self.createMode in ["polygon", "linestrip"]:
            self.line.points = [self.current[-1], self.current[0]]
        elif self.createMode == "ai_rbox":
            self.line.points = [self.current[-2], self.current[0]]
            self.current.points=self.current.points[0:3]
        elif self.createMode in ["rectangle", "ai_bbox", "line", "circle"]:
            self.current.points = self.current.points[0:1]
        elif self.createMode == "point":
            self.current = None
        self.drawingPolygon.emit(True)

    def undoLastPoint(self):
        if not self.current or self.current.isClosed():
            return
        self.current.popPoint()
        if len(self.current) > 0:
            self.line[0] = self.current[-1]
        else:
            self.current = None
            self.drawingPolygon.emit(False)
        self.update()

    def loadPixmap(self, pixmap, clear_shapes=True):
        self.pixmap = pixmap
        if not self._ai_model:
            logger.debug("Initializing AI model: SAM_B")
            model = [model for model in labelme.ai.MODELS if model.name == "SAM_B"][0]
            self._ai_model = model()

        if self._ai_model:
            self._ai_model.set_image(
                image=labelme.utils.img_qt_to_arr(self.pixmap.toImage())
            )
        if clear_shapes:
            self.shapes = []
        self.update()

    def loadShapes(self, shapes, replace=True):
        if replace:
            self.shapes = list(shapes)
            self.selectedShapes.clear()
        else:
            self.shapes.extend(shapes)
        self.storeShapes()
        self.current = None
        self.hShape = None
        self.hVertex = None
        self.hEdge = None
        self.update()

    def setShapeVisible(self, shape, value):
        self.visible[shape] = value
        self.update()

    def overrideCursor(self, cursor):
        self.restoreCursor()
        self._cursor = cursor
        QtWidgets.QApplication.setOverrideCursor(cursor)

    def restoreCursor(self):
        QtWidgets.QApplication.restoreOverrideCursor()

    def resetState(self):
        self.restoreCursor()
        self.pixmap = None
        self.shapesBackups = []
        self.update()

    def calculatePointToEdge_distance_closestpoint(self, point, edge_start, edge_end):
        edge_vector = (edge_end.x() - edge_start.x(), edge_end.y() - edge_start.y())
        edgestart_point_vector = (point.x() - edge_start.x(), point.y() - edge_start.y())
        edge_length_squared = edge_vector[0] ** 2 + edge_vector[1] ** 2
        if edge_length_squared == 0:
            # A 和 B 是同一点，返回 A
            return math.hypot(edgestart_point_vector[0], edgestart_point_vector[1]), edge_start
        # 计算投影系数 t
        t = (edge_vector[0] * edgestart_point_vector[0] + edge_vector[1] * edgestart_point_vector[
            1]) / edge_length_squared
        if t < 0:  # 垂足在 A 点左侧
            return math.hypot(edgestart_point_vector[0], edgestart_point_vector[1]), edge_start
        elif t > 1:  # 垂足在 B 点右侧
            return math.hypot(point.x() - edge_end.x(), point.y() - edge_end.y()), edge_end
        else:
            # 垂足在 AB 上
            qx = edge_start.x() + t * edge_vector[0]
            qy = edge_start.y() + t * edge_vector[1]
            return math.hypot(point.x() - qx, point.y() - qy), QtCore.QPointF(qx, qy)

    def getSegmentPoint(self, point):
        min_distance = float('inf')
        closest_point = None
        for i in range(len(self.prevhShape.points)):
            edge_start = self.prevhShape.points[i]
            edge_end = self.prevhShape.points[(i + 1) % len(self.prevhShape.points)]  # 下一个点，首尾相连
            distance, candidate_point = self.calculatePointToEdge_distance_closestpoint(point, edge_start, edge_end)

            if distance < min_distance:
                min_distance = distance
                closest_point = candidate_point

        return closest_point

    def performSegment(self):
        if len(self.segmentShape.points) == 2:
            polygon1, polygon2 = self.segment(self.prevhShape, self.getSegmentPoint(self.segmentShape.points[0]),
                                              self.getSegmentPoint(self.segmentShape.points[1]))
            self.segmentShape.points.clear()

            if len(polygon1) > 2 and len(polygon2) > 2:
                self.shapes.remove(self.prevhShape)
                self.segmentRemLabel.emit()
                self.shapesBackups.pop()

                shape1 = self.prevhShape.copy()
                shape1.points = polygon1.copy()
                shape1.point_labels = [1] * len(polygon1)
                self.shapes.append(shape1)
                self.segmentAddLabel.emit(shape1)

                shape2 = self.prevhShape.copy()
                shape2.points = polygon2.copy()
                shape2.point_labels = [1] * len(polygon2)
                self.shapes.append(shape2)
                self.segmentAddLabel.emit(shape2)

                self.storeShapes()
                self.current = None
                self.setHiding(False)
                self.update()

    def getNewPolygonFromPolygonAndPoints(self):
        if self.supplementShape.points and self.selectedShapes:
            self.selectedShapes[0].points.extend(self.supplementShape.points)
            self.selectedShapes[0].point_labels.extend(self.supplementShape.point_labels)
            self.supplementShape.points.clear()
            self.supplementShape.point_labels.clear()

            mask,newPoints = self._ai_model.predict_polygon_from_points(
                points=[[point.x(), point.y()] for point in self.selectedShapes[0].points],
                point_labels=self.selectedShapes[0].point_labels,
            )
            self.selectedShapes[0].setShapeRefined(
                points=[QtCore.QPointF(point[0], point[1]) for point in newPoints],
                point_labels=[1] * len(newPoints),
                shape_type="polygon",
            )
            self.storeShapes()
