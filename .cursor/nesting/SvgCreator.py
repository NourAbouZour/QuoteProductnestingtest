import math
from geomdl import BSpline

class SvgCreator:
    """Create SVG from Nesting Center data."""

    @staticmethod
    def createSvgNestingLayout(context, layout):
        """Convert a nesting layout into svg."""

        parts = context["Context"]["Problem"]["Parts"]
        parts_nested = layout["PartsNested"]
        raw_plate_index = layout["RawPlateIndex"]
        raw_plate = context["Context"]["Problem"]["RawPlates"][raw_plate_index]

        svg = SvgCreator.createSvgRawPlate(raw_plate)

        for part_nested in parts_nested:
            part_index = part_nested["PartIndex"]
            svg += SvgCreator.createNestedPart(parts[part_index], part_nested)

        svg += "</svg>"
        return svg

    @staticmethod
    def createSvgPart(part, width=600, height=600, margin=25, color="#0277BD"):
        """Convert a part data into svg part."""
        
        x1 = math.floor(part['Box']['X1']) - 10
        y1 = math.floor(part['Box']['Y1']) - 10
        vbWidth = math.ceil(part['Box']['X2']) - x1 + 10
        vbHeight = math.ceil(part['Box']['Y2']) - y1 + 10
        
        svg = f'<svg width="{width}" height="{height}" viewBox="{x1} {y1} {vbWidth} {vbHeight}" style="stroke:white;fill:white">'
        _path = 0

        if part.get("RectangularShape") is not None:
            svg += SvgCreator.getSvgRectangle(part, False, None, color)
        else:
            for contour in part["Contours"]:
                pathData = SvgCreator.getSvgContour(contour, True)
                if _path == 0:
                    svg += f"""<path fill="{color}" d='{pathData}'/>"""
                else:
                    svg += f"<path d='{pathData}'/>"

                _path += 1
                    
        svg += '</svg>'
        return svg

    @staticmethod
    def createSvgPart2(part, geometryInvalid = None):
        """Convert a part data into svg part."""
        
        x1 = math.floor(part['Box']['X1']) - 1
        y1 = math.floor(part['Box']['Y1']) - 1
        vbWidth = math.ceil(part['Box']['X2']) - x1 + 2
        vbHeight = math.ceil(part['Box']['Y2']) - y1 + 2
        
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x1} {y1} {vbWidth} {vbHeight}" transform = "scale(1 -1)" style="stroke:black;fill:none">'

        if part.get("RectangularShape") is not None:
            svg += SvgCreator.getSvgRectangle(part, False)
        else:
            for contour in part["Contours"]:
               path_data = SvgCreator.getSvgContour(contour, True)
               svg += f"<path d='{path_data}'/>"

            if geometryInvalid is not None:
                  for curve in geometryInvalid:
                     path_data = SvgCreator.getSvgCurve(curve, True)
                     svg += f"<path d='{path_data}' stroke='red'/>"
        
        svg += '</svg>'
        return svg

    @staticmethod
    def createNestedPart(part, part_nested):

        trans = part_nested["Transformation"]
        pos_x = trans["InsertionPt"]["X"]
        pos_y = trans["InsertionPt"]["Y"]
        angle_rad = trans["Rotation"]
        angle_deg = angle_rad * 180 / math.pi
        mirror = trans["Mirror"]
        isRectangular = part.get("RectangularShape") is not None
        ref_x = 0.0
        ref_y = 0.0

        if not isRectangular and part.get("RefPt") is not None:
            ref_x = part["RefPt"]["X"]
            ref_y = part["RefPt"]["Y"]

        if ref_x != 0 or ref_y != 0:
            x = ref_x if mirror else -ref_x
            y = -ref_y
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            x_rot = x * cos_a - y * sin_a
            y_rot = x * sin_a + y * cos_a
            pos_x += x_rot
            pos_y += y_rot

        rotate_svg = f" rotate({angle_deg})" if angle_deg != 0 else ""
        mirror_svg = " scale(-1 1)" if mirror else ""
        svg_transform = f'transform="translate({pos_x} {pos_y}){rotate_svg}{mirror_svg}"'

        svg = ""
        if part.get("RectangularShape") is not None:
            svg = SvgCreator.getSvgRectangle(part, False, svg_transform)
            return svg
        
        for contour in part["Contours"]:
            path_data = SvgCreator.getSvgContour(contour, True)
            svg += f"<path d='{path_data}' {svg_transform}/>"
    
        return svg

    def createSvgRawPlate(rp):
        x1 = 0
        y1 = 0
        length = 0
        width = 0
        svg_contour = ""

        if  rp.get("RectangularShape") is not None:
            svg_contour = SvgCreator.getSvgRectangle(rp, True)
            length = rp["RectangularShape"]["Length"]
            width = rp["RectangularShape"]["Width"]
        else:
            contours = rp["Contours"]
            for contour in contours:
                path_data = SvgCreator.getSvgContour(contour, True)
                svg_contour += f"<path d='{path_data}'/>"

            x1, y1, x2, y2 = SvgCreator.getBoundingBox(contours[0])
            length = x2 - x1
            width = y2 - y1

        x1 = math.ceil(x1) - 1
        y1 = math.ceil(y1) - 1
        vb_width = math.ceil(length) + 2
        vb_height = math.ceil(width) + 2

        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="{x1} {y1} {vb_width} {vb_height}" '
            f'transform="scale(1 -1)" '
            f'style="stroke:black;fill:none">'
        )

        svg += svg_contour
        return svg
    
    @staticmethod
    def getSvgArc(p1, p2):

        bulge = p1['B']
        dx = p2['X'] - p1['X']
        dy = p2['Y'] - p1['Y']
        c = math.sqrt(dx * dx + dy * dy)
        s = c / 2 * bulge
        r = ((c * c / 4) + s * s) / (2 * s)
        sweep =  0 if bulge < 0 else 1
        large = 1 if math.fabs(bulge) > 1 else 0
        angle = math.atan2(dy, dx) * 180 / math.pi;

        data = f" A {r},{r} {angle} {large},{sweep} {p2['X']} {p2['Y']}"
        return data
    
    @staticmethod
    def getSvgCircle(circle, moveToStart):

        cx = circle['X']
        cy = circle['Y']
        r = circle['R']
        
        data = "" if moveToStart else ""
        data += f"<circle cx {cx} cy {cy} r {r} />"
        return data

    @staticmethod
    def getSvgContour(contour: dict, move_to_start: bool):

        data = ""
        data_type = contour.get("Type")

        if data_type == "Curve2CompositeClosed":
            for curve_open in contour["Data"]["Chunks"]:
                data += SvgCreator.getSvgCurve(curve_open, move_to_start)
                move_to_start = (data == "")
            data += " Z"

        elif data_type in ("LoopBulge", "Loop"):
            data += SvgCreator.getSvgContourSimple(contour["Data"], move_to_start, True)

        elif data_type == "Circle2":
            data = SvgCreator.getSvgCircle(contour["Data"])

        elif data_type == "Ellipse2":
            data = SvgCreator.getSvgEllipse(contour["Data"])

        elif data_type is None:
            data += SvgCreator.getSvgContourSimple(contour, move_to_start, True)

        else:
            raise Exception("Unknown data type.")

        return data

    @staticmethod
    def getSvgContourSimple(contour: dict, move_to_start: bool, close_path: bool):
        vertices = contour["Vertices"]

        if len(vertices) < 2:
            return ""

        data = ("M " +SvgCreator.getPos(vertices[0])) if move_to_start else ""

        for i in range(len(vertices)):
            prev_id = i - 1 if i > 0 else len(vertices) - 1
            prev_vertex = vertices[prev_id]
            curr_vertex = vertices[i]

            if "B" in prev_vertex:
                data += SvgCreator.getSvgArc(prev_vertex, curr_vertex)
            else:
                data += " L " + SvgCreator.getPos(curr_vertex)

        # Closing arc from last to first
        if "B" in vertices[-1]:
            data += SvgCreator.getSvgArc(vertices[-1], vertices[0])

        if close_path:
            data += " Z"

        return data

    @staticmethod
    def getSvgCurve(curve: dict, move_to_start: bool):
        data = ""
        data_type = curve.get("Type")

        if data_type == "Curve2CompositeOpen":
            for curve_chunk in curve["Data"]["Chunks"]:
                data += SvgCreator.getSvgCurve(curve_chunk, move_to_start)
                move_to_start = data == ""

        elif data_type in ("PolylineBulge", "Polyline"):
            data += SvgCreator.getSvgContourSimple(curve["Data"], move_to_start, close_path=False)

        elif data_type == "EllipticalArc2":
            data = SvgCreator.getSvgEllipticalArc(curve, move_to_start)

        elif data_type == "Nurbs2":
            data = SvgCreator.getSvgSpline(curve, move_to_start)

        else:
            raise Exception("Unknown data type.")

        return data

    @staticmethod
    def getSvgEllipse(ellipse, moveToStart):
        
        cx = ellipse['Centre']['X']
        cy = ellipse['Centre']['Y']
        ax = ellipse['MajorAxis']['X']
        ay = ellipse['MajorAxis']['Y']
        ratio = ellipse['Ratio']
        rx = 0
        ry = 0
        data = ""

        if (ay == 0):
            rx = ax
            ry = ax * ratio
            data = f"<ellipse cx {cx} cy {cy} rx {rx} ry {ry} />"
        elif (ax == 0):
            rx = ay * ratio
            ry = ay
            data = f"<ellipse cx {cx} cy {cy} rx {rx} ry {ry} />"
        else:
            angle_rad = math.atan2(ay, ax)
            angle_deg = math.degrees(angle_rad)
            data = f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" transform="rotate({angle_deg} {cx} {cy})" />'
                
        return data

    @staticmethod
    def getSvgEllipticalArc(ellipse, moveToStart):

        ede = ellipse['Data']['Ellipse']
        cx = ede['Centre']['X']
        cy = ede['Centre']['Y']
        mx = ede['MajorAxis']['X']
        my = ede['MajorAxis']['Y']
        ratio = ede['Ratio']
        startAngle = ellipse['Data']['Range']['Start']
        sweepAngle = ellipse['Data']['Range']['Sweep']

        if (sweepAngle >= 360):
            sweepAngle = 359.999

        sweep = 0 if sweepAngle < 0 else 1
        large = 1 if sweepAngle > 180 else 0
        r1 = math.sqrt(mx * mx + my * my)
        r2 = r1 * ratio

        startX = r1 * math.cos(startAngle * math.pi / 180.0)
        startY = r2 * math.sin(startAngle * math.pi / 180.0)
        endX = r1 * math.cos((startAngle + sweepAngle) * math.pi / 180.0)
        endY = r2 * math.sin((startAngle + sweepAngle) * math.pi / 180.0)

        angle = math.atan2(my, mx)
        angleDeg = angle * 180 / math.pi

        s = math.sin(angle)
        c = math.cos(angle)
        sX = startX * c - startY * s + cx
        sY = startX * s + startY * c + cy
        eX = endX * c - endY * s + cx
        eY = endX * s + endY * c + cy

        data = "M " + str(sX) + "," + str(sY) if moveToStart else ""
        data += f" A {r1},{r2} {angleDeg} {large},{sweep} {eX} {eY}"
        return data
    
    @staticmethod
    def getSvgPolylineBulge(polyline, moveToStart):
        return SvgCreator.getSvgCountour(polyline['Data'], moveToStart)

    def getSvgRectangle(part, winding_cw, transform = None, color = None):
        l = part["RectangularShape"]["Length"]
        w = part["RectangularShape"]["Width"]

        if winding_cw:
            points = f"0 0 0 {w} {l} {w} {l} 0 0 0"
        else:
            points = f"0 0 {l} 0 {l} {w} 0 {w} 0 0"

        attr = ""
        if color is not None:
           attr = F'fill="{color}"'

        if transform is not None:
            attr = attr + transform;
        
        if attr != "":
           svg = f'<polyline {attr} points="{points}"/>'
        else:
           svg = f'<polyline points="{points}"/>'

        return svg
    
    @staticmethod
    def getSvgSpline(spline, moveToStart):
		
        controlPoints = spline['Data']['ControlPoints']
        cpCount = len(controlPoints)
        degree = len(spline['Data']['Knots']) - cpCount - 1

        if degree < 1:
           return ""

        data = "M " + str(controlPoints[0]['X']) + "," + str(controlPoints[0]['Y']) if moveToStart else ""
        
        if cpCount <= 4:
            data += " C"
            for i in range(1, 4):
                data += " " + str(controlPoints[i]['X']) + "," + str(controlPoints[i]['Y'])
        else:
            cp = [[pt['X'], pt['Y']] for pt in controlPoints]
            vertices = SvgCreator.getSplinePoints(cp, spline['Data']['Knots'], spline['Data']['Weights'], degree, 20)
            for i in range(1, len(vertices)):
                prevId = i - 1 if i > 0 else len(vertices) - 1
                data += " L " + str(vertices[i][0]) + " " + str(vertices[i][1])            
        
        return data

    @staticmethod   
    def getSplinePoints(cp, knots, weights, degree, n=20):
        curve = BSpline.Curve()
        curve.degree = degree
        curve.ctrlpts = cp
        curve.knotvector = knots
        curve.weights = weights if weights else [1.0] * len(cp)
        curve_points = curve.evalpts
        return curve_points

    @staticmethod 
    def getBoundingBox(path_data):
        tokens = path_data.split()
        coords_x = []
        coords_y = []
        i = 0

        while i < len(tokens):
            token = tokens[i]
            i += 1

            if token in ['M', 'L'] and (i + 1) < len(tokens):
                try:
                    coords_x.append(float(tokens[i]));
                    coords_y.append(float(tokens[i + 1]));
                    i += 2
                except ValueError:
                    break

        if not coords_x:
            return 0, 0, 1, 1

        return min(coords_x), min(coords_y), max(coords_x), max(coords_y)

    @staticmethod 
    def getPos(pos: dict):
        x = pos["X"]
        y = pos["Y"]
        return f"{x} {y}"
  