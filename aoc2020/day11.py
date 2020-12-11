import aocd
import numpy as np

TEST_DATA = """L.LL.LL.LL
LLLLLLL.LL
L.L.L..L..
LLLL.LL.LL
L.LL.LL.LL
L.LLLLL.LL
..L.L.....
LLLLLLLLLL
L.LLLLLL.L
L.LLLLL.LL"""

EMPTY = 0
SEAT = 1
OCCUPIED = 2

MAP = {".": EMPTY, "L": SEAT, "#": OCCUPIED}


def parse(data: str) -> np.ndarray:
    return np.array([[MAP[c] for c in line] for line in data.split("\n")])


def neighborhood(data: np.ndarray) -> np.ndarray:
    data = np.pad(data, ((1, 1), (1, 1)), "constant") == OCCUPIED
    return np.stack(
        [
            data[:-2, :-2],
            data[:-2, 1:-1],
            data[:-2, 2:],
            data[1:-1, :-2],
            data[1:-1, 2:],
            data[2:, :-2],
            data[2:, 1:-1],
            data[2:, 2:],
        ],
        axis=2,
    ).sum(axis=2)


def day11_part1(data: np.ndarray) -> int:
    while True:
        neighbors = neighborhood(data)
        new_data = np.where(
            (data == OCCUPIED) & (neighbors >= 4),
            SEAT,
            np.where((data == SEAT) & (neighbors == 0), OCCUPIED, data),
        )

        if np.all(new_data == data):
            return int(np.sum(new_data == OCCUPIED))
        else:
            data = new_data


def test_day11_part1():
    assert day11_part1(parse(TEST_DATA)) == 37


def neighborhood_part2(data: np.ndarray) -> np.ndarray:
    def first_occupied(line: np.ndarray) -> bool:
        (idx,) = np.where(line == OCCUPIED)
        if len(idx) == 0:
            return False
        (idx2,) = np.where(line == SEAT)
        if len(idx2) == 0:
            return True
        return idx[0] < idx2[0]

    data = np.pad(data, ((1, 1), (1, 1)), "constant")
    out = np.empty(shape=data.shape, dtype=int)
    for i in range(1, data.shape[0] - 1):
        for j in range(1, data.shape[1] - 1):
            out[i, j] = sum(
                [
                    first_occupied(data[i + 1 :, j]),
                    first_occupied(data[i, j + 1 :]),
                    first_occupied(data[i - 1 :: -1, j]),
                    first_occupied(data[i, j - 1 :: -1]),
                    first_occupied(np.diag(data[i + 1 :, j + 1 :])),
                    first_occupied(np.diag(data[i - 1 :: -1, j - 1 :: -1])),
                    first_occupied(np.diag(data[i - 1 :: -1, j + 1 :])),
                    first_occupied(np.diag(data[i + 1 :, j - 1 :: -1])),
                ]
            )
    return out[1:-1, 1:-1]


def to_string(data: np.ndarray) -> str:
    return "\n".join(
        "".join("#" if c == OCCUPIED else ("L" if c == SEAT else ".") for c in data_line)
        for data_line in data
    )


def day11_part2(data: np.ndarray) -> int:
    while True:
        neighbors = neighborhood_part2(data)
        new_data = np.where(
            (data == OCCUPIED) & (neighbors >= 5),
            SEAT,
            np.where((data == SEAT) & (neighbors == 0), OCCUPIED, data),
        )

        if np.all(new_data == data):
            return int(np.sum(new_data == OCCUPIED))
        else:
            data = new_data


def test_day11_part2():
    assert day11_part2(parse(TEST_DATA)) == 26


def main():
    data = parse(aocd.get_data(day=11, year=2020))
    print(f"day 11 part 1: {day11_part1(data)}")
    print(f"day 11 part 2: {day11_part2(data)}")


if __name__ == "__main__":
    main()
