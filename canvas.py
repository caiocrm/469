def make_grid(width, height, char=" "):
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be > 0")
    if not isinstance(char, str) or len(char) != 1:
        raise ValueError("char must be a single character string")
    return [[char for _ in range(width)] for _ in range(height)]


class Coord:
    def __init__(self, x=None, y=None, relative_to: "Coord" = None):
        self._x = 0 if x is None else x
        self._y = 0 if y is None else y
        self.relative_to = relative_to

    @property
    def x(self):
        return self._x if self.relative_to is None else self._x + self.relative_to.x

    @property
    def y(self):
        return self._y if self.relative_to is None else self._y + self.relative_to.y


class StringCanvas:
    def __init__(self, width, height, char=" "):
        self.width = width
        self.height = height
        self.char = char
        self.grid = make_grid(width, height, char)

    def _in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def write(self, coord: Coord, value=None):
        x, y = coord.x, coord.y
        if value is None:
            value = self.char
        if not isinstance(value, str) or len(value) != 1:
            raise ValueError("value must be a single character string")
        if self._in_bounds(x, y):
            self.grid[y][x] = value

    def write_line(self, coord: Coord, line):
        x0, y = coord.x, coord.y
        if not (0 <= y < self.height):
            return

        line = str(line)
        for i, ch in enumerate(line):
            x = x0 + i
            if x >= self.width:
                break
            if x >= 0:
                self.grid[y][x] = ch

    def write_column(self, coord: Coord, column):
        x, y0 = coord.x, coord.y
        if not (0 <= x < self.width):
            return

        for i, ch in enumerate(column):
            y = y0 + i
            if y >= self.height:
                break
            if y >= 0:
                ch = str(ch)
                if len(ch) != 1:
                    raise ValueError("Each column element must be a single character")
                self.grid[y][x] = ch

    def _normalize_rotation(self, rotation):
        if rotation is None:
            rotation = 0
        if rotation not in (0, 90, 180, 270, 360):
            raise ValueError("rotation must be one of: 0, 90, 180, 270, 360")
        return rotation % 360

    def _rotated_grid(self, rotation=0):
        r = self._normalize_rotation(rotation)
        g = self.grid

        if r == 0:
            return g

        if r == 90:
            return [list(row) for row in zip(*g[::-1])]

        if r == 180:
            return [row[::-1] for row in g[::-1]]

        if r == 270:
            return [list(row) for row in zip(*g)][::-1]

        return g

    def render(self, rotation=0):
        rotated = self._rotated_grid(rotation)
        return "\n".join("".join(row) for row in rotated)

    def save(self, filename, rotation=0):
        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.render(rotation=rotation))
