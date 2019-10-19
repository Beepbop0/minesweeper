extern crate rand;

use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::collections::{HashSet, VecDeque};
use std::convert::TryFrom;
use std::{io, io::Write};

fn main() {
    let model = Model::new(NUM_MINES);
    game_loop(model).expect("something went wrong handling IO");
}

fn game_loop(mut model: Model) -> io::Result<()> {
    println!("{}", model);
    io::stdout().flush()?;
    let mut buf = String::new();
    loop {
        buf.clear();
        println!("enter coordinates of where to dig");
        std::io::stdin().read_line(&mut buf)?;
        let point = match Point::try_from(buf.as_str()) {
            Ok(pt) => pt,
            Err(error) => {
                println!("{}", String::from(error));
                io::stdout().flush()?;
                continue;
            }
        };
        let update_msg = model.update(point);
        print!("{}{}", model, String::from(update_msg));
        io::stdout().flush()?;
        match update_msg {
            UpdateMsg::Lose | UpdateMsg::Win => break,
            _ => (),
        }
    }

    Ok(())
}

const ROWS: usize = 10;
const COLUMNS: usize = 10;
const NUM_MINES: usize = 10;

// type that is a valid index inside of the minefield
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Point {
    x: usize,
    y: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PointError {
    NeedXAndYCoords,
    NaN,
    InvalidRange,
}

impl TryFrom<&str> for Point {
    type Error = PointError;
    fn try_from(string: &str) -> Result<Self, Self::Error> {
        let coords: Vec<_> = string.split_whitespace().map(|x| x.parse()).collect();
        if coords.len() != 2 {
            Err(Self::Error::NeedXAndYCoords)
        } else {
            match (coords[0].clone(), coords[1].clone()) {
                // only accepts numbers in range 0 - 9
                (Ok(x), Ok(y)) => Point::try_from((x, y)),
                _ => Err(Self::Error::NaN),
            }
        }
    }
}

impl TryFrom<(usize, usize)> for Point {
    type Error = PointError;
    fn try_from((x, y): (usize, usize)) -> Result<Self, Self::Error> {
        if x < ROWS && y < COLUMNS {
            Ok(Self { x, y })
        } else {
            Err(Self::Error::InvalidRange)
        }
    }
}

impl From<Point> for (usize, usize) {
    fn from(pt: Point) -> Self {
        (pt.x, pt.y)
    }
}

impl std::fmt::Display for PointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let output: String = self.clone().into();
        write!(f, "{}", output)
    }
}

impl From<PointError> for String {
    fn from(err: PointError) -> Self {
        String::from(match err {
            PointError::NeedXAndYCoords | PointError::NaN => {
                "Supply two numbers, seperated by a space"
            }
            PointError::InvalidRange => "Supply two numbers with values between 0 and 9",
        })
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum UpdateMsg {
    Lose,
    Win,
    PreviouslyDug,
    Continue,
}

impl From<UpdateMsg> for String {
    fn from(msg: UpdateMsg) -> Self {
        String::from(match msg {
            UpdateMsg::Lose => "Yikes! hit a 💣\n",
            UpdateMsg::Win => "🎉 you won!\n",
            UpdateMsg::PreviouslyDug => "That position is already visible\n",
            UpdateMsg::Continue => "",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GameEntity {
    Mine,
    NotVisited,
    Empty(u8),
}

impl From<GameEntity> for String {
    fn from(entity: GameEntity) -> Self {
        match entity {
            GameEntity::Empty(val) => val.to_string(),
            _ => "*".to_string(),
        }
    }
}

#[derive(Debug)]
pub struct Model {
    minefield: Vec<Vec<GameEntity>>,
}

// used for mocking boards
impl From<Vec<Vec<GameEntity>>> for Model {
    fn from(minefield: Vec<Vec<GameEntity>>) -> Self {
        Self { minefield }
    }
}

impl std::fmt::Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let output = String::from(self);
        write!(f, "{}", output)
    }
}

impl From<&Model> for String {
    fn from(model: &Model) -> Self {
        let top_bottom_border = std::iter::repeat('═')
            .take((COLUMNS * 2) - 1)
            .collect::<String>();
        let mut display_rows = vec![String::new(); COLUMNS];
        // transpose matrix to display rows along x axis and columns along y
        for col in 0..COLUMNS {
            for row in 0..ROWS {
                display_rows[col]
                    .push_str(&format!("{} ", &String::from(model.minefield[row][col])));
            }
        }
        let mut display_rows = display_rows
            .into_iter()
            .enumerate()
            .rev()
            .map(|(i, row)| format!("{} ║{}║\n", i, row.trim_end()))
            .collect::<Vec<_>>();
        display_rows.insert(0, format!("  ╔{}╗\n", &top_bottom_border));
        display_rows.push(format!("  ╚{}╝\n", &top_bottom_border));
        display_rows.push(format!(
            "   {}\n",
            (0..COLUMNS)
                .map(|j| format!("{} ", j))
                .collect::<String>()
                .trim_end()
        ));
        display_rows.into_iter().collect()
    }
}

// Generates NUM_MINES unique coordinates for where the mines will be located
fn gen_mine_indices(num_mines: usize) -> Vec<(usize, usize)> {
    let mut rng = SmallRng::from_entropy();
    let mut indices = HashSet::with_capacity(NUM_MINES);
    while indices.len() < num_mines {
        let x = rng.gen_range(0, ROWS);
        let y = rng.gen_range(0, COLUMNS);
        if !indices.contains(&(x, y)) {
            indices.insert((x, y));
        }
    }
    indices.into_iter().collect()
}

impl Model {
    fn new(num_mines: usize) -> Self {
        let mut minefield = vec![vec![GameEntity::NotVisited; COLUMNS]; ROWS];
        let indices = gen_mine_indices(num_mines);
        for (x, y) in indices {
            minefield[x][y] = GameEntity::Mine;
        }
        Self { minefield }
    }

    fn get(&self, point: Point) -> &GameEntity {
        &self.minefield[point.x][point.y]
    }

    fn get_mut(&mut self, point: Point) -> &mut GameEntity {
        &mut self.minefield[point.x][point.y]
    }

    fn visit(&mut self, point: Point, mine_count: u8) {
        *self.get_mut(point) = GameEntity::Empty(mine_count);
    }

    pub fn update(&mut self, point: Point) -> UpdateMsg {
        match self.get(point) {
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
        }
    }

    fn exists_winner(&self) -> bool {
        self.minefield
            .iter()
            .all(|row| row.iter().all(|elem| *elem != GameEntity::NotVisited))
    }

    // basic BFS algo to show all empty cells with no neighboring bombs, and cells that do neighbor bombs
    // NOTE: there is no explicit visited list. Updating the minecount means that the node has been visited
    fn flood_fill(&mut self, start: Point) {
        let mut queue = VecDeque::new();
        self.visit(start, self.num_neighboring_mines(start));
        queue.push_back(start);

        while let Some(point) = queue.pop_front() {
            let count = self.num_neighboring_mines(point);
            self.visit(point, count);
            // DO NOT add neighbors that are near a mine
            if count == 0 {
                for neighbor in neighbor_cell_indices(point) {
                    if self.have_not_visited(neighbor) {
                        queue.push_back(neighbor);
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
        *self.get(point) == GameEntity::Mine
    }

    fn have_not_visited(&self, point: Point) -> bool {
        *self.get(point) == GameEntity::NotVisited
    }
}

// produces all valid indices that are 1 hop away from the present point
fn neighbor_cell_indices(
    Point {
        x: start_x,
        y: start_y,
    }: Point,
) -> impl Iterator<Item = Point> {
    // ensure we don't panic on subtracting from 0
    let x_min = start_x.checked_sub(1).unwrap_or(0);
    let y_min = start_y.checked_sub(1).unwrap_or(0);

    (x_min..=(start_x + 1))
        .flat_map(move |x| (y_min..=(start_y + 1)).map(move |y| (x, y)))
        .filter(move |(x, y)| *x != start_x || *y != start_y)
        .filter_map(|(x, y)| Point::try_from((x, y)).ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn land_on_already_visited_sends_prev_dug() {
        let mut grid = vec![vec![GameEntity::NotVisited; COLUMNS]; ROWS];
        grid[4][5] = GameEntity::Empty(0);
        let mut model = Model::from(grid);
        let point = Point::try_from((4, 5)).unwrap();
        let msg = model.update(point);
        assert_eq!(msg, UpdateMsg::PreviouslyDug);
    }

    #[test]
    fn land_on_not_visited_sends_continue() {
        let mut grid = vec![vec![GameEntity::NotVisited; COLUMNS]; ROWS];
        grid[0][0] = GameEntity::Mine;
        let mut model = Model::from(grid);
        let point = Point::try_from((1, 1)).unwrap();
        let msg = model.update(point);
        assert_eq!(msg, UpdateMsg::Continue);
    }

    #[test]
    fn land_on_mine_sends_lose_msg() {
        let mut grid = vec![vec![GameEntity::NotVisited; COLUMNS]; ROWS];
        grid[3][4] = GameEntity::Mine;
        let mut model = Model::from(grid);
        let point = Point::try_from((3, 4)).unwrap();
        let msg = model.update(point);
        assert_eq!(msg, UpdateMsg::Lose);
    }

    #[test]
    fn game_with_one_center_mine() {
        let mut grid = vec![vec![GameEntity::NotVisited; COLUMNS]; ROWS];
        grid[4][4] = GameEntity::Mine;
        let mut model = Model::from(grid);
        let point = Point::try_from((0, 0)).unwrap();
        model.flood_fill(point);
        for y in 0..COLUMNS {
            for x in 0..ROWS {
                if y < 3 || y > 5 || x < 3 || x > 5 {
                    assert_eq!(model.minefield[y][x], GameEntity::Empty(0));
                } else if y != 4 || x != 4 {
                    assert_eq!(model.minefield[y][x], GameEntity::Empty(1));
                }
            }
        }

        assert!(model.exists_winner());
    }

    #[test]
    fn flood_fill_corner_one_mine() {
        let mut grid = vec![vec![GameEntity::NotVisited; COLUMNS]; ROWS];
        grid[0][1] = GameEntity::Mine;
        let mut model = Model::from(grid);
        let point = Point::try_from((0, 0)).unwrap();
        model.flood_fill(point);
        assert_eq!(model.minefield[0][0], GameEntity::Empty(1));
        assert!(model.minefield[1]
            .iter()
            .skip(1)
            .all(|entity| *entity == GameEntity::NotVisited));
        assert!(model
            .minefield
            .iter()
            .skip(1)
            .all(|row| row.iter().all(|entity| *entity == GameEntity::NotVisited)));
        assert!(!model.exists_winner());
    }

    #[test]
    fn flood_fill_corner() {
        let mut grid = vec![vec![GameEntity::NotVisited; COLUMNS]; ROWS];
        grid[1][0] = GameEntity::Mine;
        grid[1][1] = GameEntity::Mine;
        grid[0][1] = GameEntity::Mine;
        let mut model = Model::from(grid);
        let point = Point::try_from((0, 0)).unwrap();
        model.flood_fill(point);
        assert_eq!(model.minefield[0][0], GameEntity::Empty(3));
        for i in 0..2 {
            assert!(model.minefield[i]
                .iter()
                .skip(2)
                .all(|x| *x == GameEntity::NotVisited));
        }
        assert!(model
            .minefield
            .iter()
            .skip(3)
            .all(|row| row.iter().all(|x| *x == GameEntity::NotVisited)));
    }

    #[test]
    fn mine_count_sparse() {
        let mut grid = vec![vec![GameEntity::NotVisited; 3]; 3];
        grid[0][0] = GameEntity::Mine;
        grid[2][2] = GameEntity::Mine;
        grid[2][1] = GameEntity::Mine;
        let model = Model::from(grid);
        let point = Point::try_from((1, 1)).unwrap();
        assert_eq!(model.num_neighboring_mines(point), 3);
    }

    #[test]
    fn neighbor_indices_for_corners() {
        // top left corner
        let point = Point::try_from((0, 0)).unwrap();
        let mut iter = neighbor_cell_indices(point).map(Point::into);
        assert_eq!(iter.next(), Some((0, 1)));
        assert_eq!(iter.next(), Some((1, 0)));
        assert_eq!(iter.next(), Some((1, 1)));
        assert_eq!(iter.next(), None);

        // bottom left corner
        let point = Point::try_from((0, COLUMNS - 1)).unwrap();
        let mut iter = neighbor_cell_indices(point).map(Point::into);
        assert_eq!(iter.next(), Some((0, COLUMNS - 2)));
        assert_eq!(iter.next(), Some((1, COLUMNS - 2)));
        assert_eq!(iter.next(), Some((1, COLUMNS - 1)));
        assert_eq!(iter.next(), None);

        // top right corner
        let point = Point::try_from((ROWS - 1, 0)).unwrap();
        let mut iter = neighbor_cell_indices(point).map(Point::into);
        assert_eq!(iter.next(), Some((ROWS - 2, 0)));
        assert_eq!(iter.next(), Some((ROWS - 2, 1)));
        assert_eq!(iter.next(), Some((ROWS - 1, 1)));
        assert_eq!(iter.next(), None);

        // bottom right corner
        let point = Point::try_from((ROWS - 1, COLUMNS - 1)).unwrap();
        let mut iter = neighbor_cell_indices(point).map(Point::into);
        assert_eq!(iter.next(), Some((ROWS - 2, COLUMNS - 2)));
        assert_eq!(iter.next(), Some((ROWS - 2, COLUMNS - 1)));
        assert_eq!(iter.next(), Some((ROWS - 1, COLUMNS - 2)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn neighbor_indices_for_middlepoint() {
        let point = Point::try_from((1, 3)).unwrap();
        let mut iter = neighbor_cell_indices(point).map(Point::into);
        assert_eq!(iter.next(), Some((0, 2)));
        assert_eq!(iter.next(), Some((0, 3)));
        assert_eq!(iter.next(), Some((0, 4)));
        assert_eq!(iter.next(), Some((1, 2)));
        assert_eq!(iter.next(), Some((1, 4)));
        assert_eq!(iter.next(), Some((2, 2)));
        assert_eq!(iter.next(), Some((2, 3)));
        assert_eq!(iter.next(), Some((2, 4)));
        assert_eq!(iter.next(), None);
    }
}
