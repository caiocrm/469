import pandas as pd
from canvas import StringCanvas, Coord


if __name__ == "__main__":
    books = pd.read_csv("data/raw_books_by_line.csv", dtype=str)
    books_text = books["line"].tolist()
    line_length = max([len(book) for book in books_text])
    print(len(books_text))
    patterns = ["1", "469", "13", "49", "2", "3", "4", "5", "6", "7", "8", "9"]
    canvas = StringCanvas(line_length*len(patterns) + 20, len(books_text), " ")
    def normalize_book(book, keep, fixed_char=None):
        final_str = ""
        for c in book.strip().lower():
            if c in keep:
                if fixed_char is not None:
                    final_str += fixed_char
                else:
                    final_str += c
            else:
                final_str += " "

        # final_str = final_str.replace(f"11", f" 1")
        return final_str

    origx = 0
    origy = 0
    curry = 0
    rotation = 360
    for idx, row in books.groupby(["book_idx"]):
        book = row["line"].tolist()
        origin = Coord(origx, origy)
        
        for line in book:
            line = line.ljust(line_length, " ")
            currx = 0
            for offset in range(0, len(patterns)):
                canvas.write_line(Coord(currx, curry, origin), normalize_book(line, patterns[offset]))
                currx += len(line)
                canvas.write(Coord(currx, curry, origin), "|")
                currx += 2
            curry += 1
        canvas.write_line(Coord(0, curry, origin), "-"*currx)
        curry += 1
    canvas.save("grid.txt", rotation=rotation)
    print(canvas.render(rotation=rotation))
