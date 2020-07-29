use fastrand::Rng;
use std::collections::{HashSet, VecDeque};
use std::fmt;
use std::io::{self, Write};
use std::ops::{Index, IndexMut};

const ROWS: usize = 10;
const COLUMNS: usize = 10;
const NUM_MINES: usize = 10;

type Point = (usize, usize);

fn main() -> io::Result<()> {
    let mut model = Model::new(NUM_MINES);
    println!("{}", model);
    io::stdout().flush()?;
    let mut buf = String::new();
    loop {
        buf.clear();
        println!("enter coordinates of where to dig");
        io::stdin().read_line(&mut buf)?;
        let point = match get_user_point(&buf) {
            Some(pt) => pt,
            _ => {
                println!("Proivde an x and y coordinate with a space between them");
                io::stdout().flush()?;
                continue;
            }
        };
        let try_update_msg = model.update(point);
        match try_update_msg {
            None => println!("Enter valid coordinates of 0-9"),
            Some(update_msg) => {
                print!("{}\n{}", model, update_msg);
                io::stdout().flush()?;
                if let UpdateMsg::Lose | UpdateMsg::Win = update_msg {
                    break;
                }
            }
        }
    }

    Ok(())
}

fn get_user_point(s: &str) -> Option<Point> {
    let mut parser = s.split_whitespace().filter_map(|input| input.parse().ok());
    match (parser.next(), parser.next()) {
        (Some(x), Some(y)) if x < COLUMNS && y < ROWS => Some((x, y)),
        _ => None,
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum UpdateMsg {
    Lose,
    Win,
    PreviouslyDug,
    Continue,
}

impl fmt::Display for UpdateMsg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lose => write!(f, "Yikes! hit a 💣"),
            Self::Win => write!(f, "🎉 you won!"),
            Self::PreviouslyDug => write!(f, "That position is already visible"),
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameEntity {
    Mine,
    NotVisited,
    Empty(u8),
}

impl fmt::Display for GameEntity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GameEntity::Empty(val) => write!(f, "{}", val),
            _ => write!(f, "*"),
        }
    }
}

pub struct Model {
    minefield: [GameEntity; ROWS * COLUMNS],
}

// used for mocking boards
impl From<[GameEntity; ROWS * COLUMNS]> for Model {
    fn from(minefield: [GameEntity; ROWS * COLUMNS]) -> Self {
        Self { minefield }
    }
}

impl fmt::Debug for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in (0..ROWS).rev() {
            let base_index = row * COLUMNS;
            write!(
                f,
                "{:?}\n",
                &self.minefield[base_index..(base_index + COLUMNS)]
            )?;
        }
        Ok(())
    }
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in (0..ROWS).rev() {
            write!(f, "{}  ", row)?;
            for col in 0..COLUMNS {
                write!(f, "{} ", self[(col, row)])?;
            }
            writeln!(f)?;
        }
        write!(f, "y\n x ")?;
        for col in 0..COLUMNS {
            write!(f, "{} ", col)?;
        }
        writeln!(f)?;

        Ok(())
    }
}

// Generates NUM_MINES unique coordinates for where the mines will be located
fn gen_mine_indices(num_mines: usize) -> Vec<(usize, usize)> {
    let rng = Rng::new();
    let mut indices = HashSet::with_capacity(NUM_MINES);
    while indices.len() < num_mines {
        let pt = (rng.usize(0..ROWS), rng.usize(0..COLUMNS));
        if !indices.contains(&pt) {
            indices.insert(pt);
        }
    }
    indices.into_iter().collect()
}

impl Index<Point> for Model {
    type Output = GameEntity;
    fn index(&self, index: Point) -> &Self::Output {
        self.get(index).unwrap()
    }
}

impl IndexMut<Point> for Model {
    fn index_mut(&mut self, index: Point) -> &mut Self::Output {
        self.get_mut(index).unwrap()
    }
}
impl Model {
    fn new(num_mines: usize) -> Self {
        let mut model = Self {
            minefield: [GameEntity::NotVisited; ROWS * COLUMNS],
        };
        let indices = gen_mine_indices(num_mines);
        for (x, y) in indices {
            if let Some(entity) = model.get_mut((x, y)) {
                *entity = GameEntity::Mine;
            }
        }
        model
    }

    fn get(&self, (x, y): Point) -> Option<&GameEntity> {
        // y is row, x is column. y multiplied by the number of columns in a row
        self.minefield.get((y * COLUMNS) + x)
    }

    fn get_mut(&mut self, (x, y): Point) -> Option<&mut GameEntity> {
        self.minefield.get_mut((y * COLUMNS) + x)
    }

    pub fn update(&mut self, point: Point) -> Option<UpdateMsg> {
        Some(match self.get(point)? {
            GameEntity::Mine => UpdateMsg::Lose,
            GameEntity::NotVisited => {
                self.flood_fill(point);
                if self.exists_winner() {
                    UpdateMsg::Win
                } else {
                    UpdateMsg::Continue
                }
            }
            GameEntity::Empty(_) => UpdateMsg::PreviouslyDug,
        })
    }

    fn exists_winner(&self) -> bool {
        self.minefield
            .iter()
            .all(|entity| *entity != GameEntity::NotVisited)
    }

    // basic BFS algo to show all empty cells with no neighboring bombs, and cells that do neighbor bombs
    fn flood_fill(&mut self, start: Point) {
        if self[start] == GameEntity::NotVisited {
            let mut queue = VecDeque::new();
            queue.push_back(start);

            while let Some(point) = queue.pop_front() {
                let count = self.num_neighboring_mines(point);
                self[point] = GameEntity::Empty(count);
                if count == 0 {
                    for n in neighbor_cell_indices(point) {
                        if self[n] == GameEntity::NotVisited && !queue.contains(&n) {
                            queue.push_back(n);
                        }
                    }
                }
            }
        }
    }

    fn num_neighboring_mines(&self, point: Point) -> u8 {
        neighbor_cell_indices(point).fold(
            0,
            |acc, point| if self.is_mine(point) { acc + 1 } else { acc },
        )
    }

    fn is_mine(&self, point: Point) -> bool {
        self.get(point)
            .map(|cell| *cell == GameEntity::Mine)
            .unwrap_or(false)
    }
}

// produces all valid indices that are 1 hop away from the present point
fn neighbor_cell_indices((start_x, start_y): Point) -> impl Iterator<Item = Point> {
    // ensure we don't panic on subtracting from 0 on an unsigned type
    let (x_min, x_max) = (start_x.saturating_sub(1), (start_x + 1).max(COLUMNS - 1));
    let (y_min, y_max) = (start_y.saturating_sub(1), (start_y + 1).max(ROWS - 1));

    (x_min..=(start_x + 1).min(COLUMNS - 1))
        .flat_map(move |x| (y_min..=(start_y + 1).min(ROWS - 1)).map(move |y| (x, y)))
        .filter(move |(x, y)| *x != start_x || *y != start_y)
        .map(Point::from)
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    fn get_mut(grid: &mut [GameEntity], (x, y): Point) -> &mut GameEntity {
        &mut grid[(y * COLUMNS) + x]
    }

    fn set_mine(grid: &mut [GameEntity], pt: Point) {
        *get_mut(grid, pt) = GameEntity::Mine;
    }

    #[test]
    fn model_generates_correct_num_mines() {
        let num_mines = 22;
        assert!(num_mines < COLUMNS * ROWS);
        let model = Model::new(num_mines);
        assert_eq!(
            model
                .minefield
                .iter()
                .fold(0, |acc, x| if *x == GameEntity::Mine {
                    acc + 1
                } else {
                    acc
                }),
            num_mines
        );
    }

    #[test]
    fn land_on_already_visited_sends_prev_dug() {
        let mut grid = [GameEntity::NotVisited; ROWS * COLUMNS];
        *get_mut(&mut grid, (4, 5)) = GameEntity::Empty(0);
        let mut model = Model::from(grid);
        let point = (4, 5);
        let msg = model.update(point);
        assert_eq!(msg, Some(UpdateMsg::PreviouslyDug));
    }

    #[test]
    fn land_on_not_visited_sends_continue() {
        let mut grid = [GameEntity::NotVisited; ROWS * COLUMNS];
        set_mine(&mut grid, (0, 0));
        let mut model = Model::from(grid);
        let point = (1, 1);
        let msg = model.update(point);
        assert_eq!(msg, Some(UpdateMsg::Continue));
    }

    #[test]
    fn land_on_mine_sends_lose_msg() {
        let mut grid = [GameEntity::NotVisited; ROWS * COLUMNS];
        set_mine(&mut grid, (3, 4));
        let mut model = Model::from(grid);
        let point = (3, 4);
        let msg = model.update(point);
        assert_eq!(msg, Some(UpdateMsg::Lose));
    }

    #[test]
    fn game_with_one_center_mine() {
        let mut grid = [GameEntity::NotVisited; ROWS * COLUMNS];
        set_mine(&mut grid, (4, 4));
        let mut model = Model::from(grid);
        let point = (0, 0);
        model.flood_fill(point);
        for y in 0..COLUMNS {
            for x in 0..ROWS {
                if y < 3 || y > 5 || x < 3 || x > 5 {
                    assert_eq!(model[(x, y)], GameEntity::Empty(0));
                } else if y != 4 && x != 4 {
                    assert_eq!(model[(x, y)], GameEntity::Empty(1));
                }
            }
        }

        assert!(model.exists_winner());
    }

    #[test]
    fn flood_fill_corner_one_mine() {
        let mut grid = [GameEntity::NotVisited; ROWS * COLUMNS];
        // grid[0][1] = GameEntity::Mine;
        set_mine(&mut grid, (0, 1));
        let mut model = Model::from(grid);
        let point = (0, 0);
        model.flood_fill(point);
        dbg!(&model);
        assert_eq!(model[(0, 0)], GameEntity::Empty(1));
        // first two rows should all have not visisted, except for the first item
        for row in 0..2 {
            let start_of_row = row * COLUMNS;
            assert!(
                model.minefield[(start_of_row + 1)..(start_of_row + COLUMNS)]
                    .iter()
                    .skip(1)
                    .all(|entity| *entity == GameEntity::NotVisited)
            );
        }

        assert!(model
            .minefield
            .iter()
            .skip(2 * COLUMNS)
            .all(|entity| *entity == GameEntity::NotVisited));
        assert!(!model.exists_winner());
    }

    #[test]
    fn flood_fill_corner() {
        let mut grid = [GameEntity::NotVisited; ROWS * COLUMNS];
        set_mine(&mut grid, (1, 0));
        set_mine(&mut grid, (1, 1));
        set_mine(&mut grid, (0, 1));
        let mut model = Model::from(grid);
        let point = (0, 0);
        model.flood_fill(point);
        assert_eq!(model[(0, 0)], GameEntity::Empty(3));
        for row in 0..2 {
            assert!(
                model.minefield[((row * COLUMNS) + 2)..((row + 1) * COLUMNS)]
                    .iter()
                    .all(|x| *x == GameEntity::NotVisited)
            );
        }
        assert!(model.minefield[(2 * COLUMNS)..]
            .iter()
            .skip(3)
            .all(|entity| *entity == GameEntity::NotVisited));
    }

    // example of wrong minecount:
    // 9  0 1 @ 1 0 0 0 0 0 0
    // 8  0 1 1 1 0 1 1 1 0 0
    // 7  0 0 0 0 0 1 @ 1 0 0
    // 6  0 0 0 0 0 1 * 1 0 0
    // 5  2 1 1 0 0 1 * 2 1 0
    // 4  * @ 1 0 0 1 @ @ 2 1
    // 3  * 1 1 0 1 3 * * * @
    // 2  * 1 0 0 1 @ @ * * *
    // 1  @ 1 0 0 1 * * * * *
    // 0  2 @ * * * * * * * 1
    // y
    //  x 0 1 2 3 4 5 6 7 8 9
    // the count at (9, 0) should be 0 and should flood fill.
    #[test]
    fn regression_test_1() {
        let mut grid = [GameEntity::NotVisited; ROWS * COLUMNS];
        for pt in &[
            (0, 1),
            (1, 0),
            (1, 4),
            (2, 9),
            (5, 2),
            (6, 2),
            (6, 4),
            (6, 7),
            (7, 4),
            (9, 3),
        ] {
            set_mine(&mut grid, *pt);
        }
        let mut model = Model::from(grid);
        model.update((9, 0));
        assert_eq!(model[(9, 0)], GameEntity::Empty(0));
    }

    #[test]
    fn neighbor_works_in_corners() {
        assert!(
            neighbor_cell_indices((0, 0)).all(|pt| pt == (0, 1) || pt == (1, 1) || pt == (1, 0))
        );
        assert!(neighbor_cell_indices((COLUMNS - 1, 0))
            .all(|pt| pt == (COLUMNS - 2, 0) || pt == (COLUMNS - 1, 1) || pt == (COLUMNS - 2, 1)));
        assert!(neighbor_cell_indices((0, ROWS - 1))
            .all(|pt| pt == (0, ROWS - 2) || pt == (1, ROWS - 1) || pt == (1, ROWS - 2)));
        assert!(neighbor_cell_indices((COLUMNS - 1, ROWS - 1))
            .all(|pt| pt == (COLUMNS - 2, ROWS - 1)
                || pt == (COLUMNS - 2, ROWS - 2)
                || pt == (COLUMNS - 1, ROWS - 2)));
    }
}
