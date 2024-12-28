use core::fmt;
use std::env;
use std::error::Error;
use std::fmt::Debug;
use std::fs;
use std::io::{self, Write};

#[derive(Debug, Clone)]
enum Value {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Null,
    Nil,
}
impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Str(val) => write!(f, "{}", val),
            Value::Int(val) => write!(f, "{}", val),
            Value::Float(val) => write!(f, "{:?}", val),
            Value::Bool(val) => write!(f, "{}", if *val { "true" } else { "false" }),
            Value::Null => write!(f, "{}", String::from("null")),
            Value::Nil => write!(f, "{}", String::from("nil")),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
enum TokenType {
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    Comma,
    Period,
    Minus,
    Plus,
    Semicolon,
    Slash,
    Star,

    Bang,
    BangEqual,
    Equal,
    EqualEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,

    Identifier,
    Str,
    Number,

    And,
    Class,
    Else,
    False,
    Fun,
    For,
    If,
    Nil,
    Or,
    Print,
    Return,
    Super,
    This,
    True,
    Var,
    While,

    Eof,
}
impl TokenType {
    pub fn to_string(self) -> String {
        let val = match self {
            TokenType::LeftParen => "LEFT_PAREN",
            TokenType::RightParen => "RIGHT_PAREN",
            TokenType::LeftBrace => "LEFT_BRACE",
            TokenType::RightBrace => "RIGHT_BRACE",
            TokenType::Comma => "COMMA",
            TokenType::Period => "DOT",
            TokenType::Minus => "MINUS",
            TokenType::Plus => "PLUS",
            TokenType::Semicolon => "SEMICOLON",
            TokenType::Slash => "SLASH",
            TokenType::Star => "STAR",
            TokenType::Bang => "BANG",
            TokenType::BangEqual => "BANG_EQUAL",
            TokenType::Equal => "EQUAL",
            TokenType::EqualEqual => "EQUAL_EQUAL",
            TokenType::Greater => "GREATER",
            TokenType::GreaterEqual => "GREATER_EQUAL",
            TokenType::Less => "LESS",
            TokenType::LessEqual => "LESS_EQUAL",
            TokenType::Identifier => "IDENTIFIER",
            TokenType::Str => "STRING",
            TokenType::Number => "NUMBER",
            TokenType::And => "AND",
            TokenType::Class => "CLASS",
            TokenType::Else => "ELSE",
            TokenType::False => "FALSE",
            TokenType::Fun => "FUN",
            TokenType::For => "FOR",
            TokenType::If => "IF",
            TokenType::Nil => "NIL",
            TokenType::Or => "OR",
            TokenType::Print => "PRINT",
            TokenType::Return => "RETURN",
            TokenType::Super => "SUPER",
            TokenType::This => "THIS",
            TokenType::True => "TRUE",
            TokenType::Var => "VAR",
            TokenType::While => "WHILE",
            TokenType::Eof => "EOF",
        };

        String::from(val)
    }
}

#[derive(Debug, Clone)]
enum LexerError {
    ParseError(usize, usize),
    UnexpectedToken(usize, usize, char),
    UnterminatedString(usize, usize),
}

impl fmt::Display for LexerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LexerError::ParseError(line, _) => write!(f, "[line {}] Error: Parse Error", line),
            LexerError::UnexpectedToken(line, _, ch) => {
                write!(f, "[line {}] Error: Unexpected character: {}", line, ch)
            }
            LexerError::UnterminatedString(line, _) => {
                write!(f, "[line {}] Error: Unterminated string.", line)
            }
        }
    }
}

impl std::error::Error for LexerError {
    fn description(&self) -> &str {
        "lexer error"
    }
}

#[derive(Debug, Clone)]
struct Token {
    token_type: TokenType,
    loxme: String,
    value: Option<Value>,
}
impl Token {
    pub fn new(token_type: TokenType, loxme: String, value: Option<Value>) -> Self {
        Self {
            token_type,
            loxme,
            value,
        }
    }
    pub fn log(self) {
        let token_type = self.token_type.to_string();
        let value = match self.value {
            Some(val) => val,
            None => Value::Null,
        };
        println!("{} {} {}", token_type, self.loxme, value);
    }
}

impl PartialEq<TokenType> for Token {
    fn eq(&self, other: &TokenType) -> bool {
        self.token_type == *other
    }
}

impl PartialEq<Token> for TokenType {
    fn eq(&self, other: &Token) -> bool {
        *self == other.token_type
    }
}

#[derive(Debug)]
struct Lexer {
    tokens: Vec<Token>,
    input: String,
    index: usize,
    line: usize,
    pos: usize,
    errors: Vec<LexerError>,
}

impl Lexer {
    pub fn new(input: String) -> Self {
        Self {
            tokens: Vec::new(),
            input,
            index: 0,
            line: 1,
            pos: 0,
            errors: Vec::<LexerError>::new(),
        }
    }

    fn take_whitespace(&mut self) {
        while let Some(ch) = self.input.chars().nth(self.index) {
            if ch == '\n' {
                self.line += 1;
                self.pos = 0;
            }

            if ch.is_whitespace() {
                self.index += 1;
            } else {
                break;
            }
        }
    }

    fn report_error(&mut self, err: LexerError) {
        writeln!(io::stderr(), "{}", err).unwrap();
        self.errors.push(err);
    }

    fn has_error(&self) -> bool {
        self.errors.len() > 0
    }

    fn peek(&self) -> Option<char> {
        self.input.chars().nth(self.index + 1)
    }

    fn take_until(&mut self, ch: char) {
        while let Some(curr) = self.peek() {
            if ch != curr {
                self.index += 1;
                self.pos += 1;
                continue;
            }

            break;
        }
    }

    fn collect_until(&mut self, ch: char) -> Option<String> {
        let from = self.index;
        while let Some(curr) = self.peek() {
            self.index += 1;
            self.pos += 1;

            if ch != curr {
                continue;
            }

            return Some(self.input[from..self.index + 1].to_string());
        }

        None
    }

    fn take_number(&mut self) -> String {
        let from = self.index;
        while let Some(ch) = self.peek() {
            if ch != '.' && !ch.is_numeric() {
                break;
            }

            self.index += 1;
            self.pos += 1;
        }

        self.input[from..self.index + 1].to_string()
    }

    fn take_identifier(&mut self) -> String {
        let from = self.index;
        while let Some(ch) = self.peek() {
            if ch != '_' && !ch.is_alphanumeric() {
                break;
            }

            self.index += 1;
            self.pos += 1;
        }

        self.input[from..self.index + 1].to_string()
    }

    pub fn tokenize(&mut self) -> Result<(), LexerError> {
        let strlen = self.input.len();

        while self.index < strlen {
            self.take_whitespace();
            let token = match self.input.chars().nth(self.index) {
                None => {
                    if (self.tokens.last().is_some_and(|x| *x == TokenType::Eof)) {
                        None
                    } else {
                        Some(Token::new(
                            TokenType::Eof,
                            String::from(""),
                            Some(Value::Null),
                        ))
                    }
                }
                Some('(') => Some(Token::new(
                    TokenType::LeftParen,
                    String::from("("),
                    Some(Value::Null),
                )),
                Some(')') => Some(Token::new(
                    TokenType::RightParen,
                    String::from(")"),
                    Some(Value::Null),
                )),
                Some('{') => Some(Token::new(
                    TokenType::LeftBrace,
                    String::from("{"),
                    Some(Value::Null),
                )),
                Some('}') => Some(Token::new(
                    TokenType::RightBrace,
                    String::from("}"),
                    Some(Value::Null),
                )),
                Some('*') => Some(Token::new(
                    TokenType::Star,
                    String::from("*"),
                    Some(Value::Null),
                )),
                Some('.') => Some(Token::new(
                    TokenType::Period,
                    String::from("."),
                    Some(Value::Null),
                )),
                Some(',') => Some(Token::new(
                    TokenType::Comma,
                    String::from(","),
                    Some(Value::Null),
                )),
                Some('+') => Some(Token::new(
                    TokenType::Plus,
                    String::from("+"),
                    Some(Value::Null),
                )),
                Some('-') => Some(Token::new(
                    TokenType::Minus,
                    String::from("-"),
                    Some(Value::Null),
                )),
                Some(';') => Some(Token::new(
                    TokenType::Semicolon,
                    String::from(";"),
                    Some(Value::Null),
                )),
                Some('=') => {
                    if Some('=') == self.peek() {
                        self.index += 1;
                        self.pos += 1;
                        Some(Token::new(
                            TokenType::EqualEqual,
                            String::from("=="),
                            Some(Value::Null),
                        ))
                    } else {
                        Some(Token::new(
                            TokenType::Equal,
                            String::from("="),
                            Some(Value::Null),
                        ))
                    }
                }
                Some('!') => {
                    if Some('=') == self.peek() {
                        self.index += 1;
                        self.pos += 1;
                        Some(Token::new(
                            TokenType::BangEqual,
                            String::from("!="),
                            Some(Value::Null),
                        ))
                    } else {
                        Some(Token::new(
                            TokenType::Bang,
                            String::from("!"),
                            Some(Value::Null),
                        ))
                    }
                }
                Some('>') => {
                    if Some('=') == self.peek() {
                        self.index += 1;
                        self.pos += 1;
                        Some(Token::new(
                            TokenType::GreaterEqual,
                            String::from(">="),
                            Some(Value::Null),
                        ))
                    } else {
                        Some(Token::new(
                            TokenType::Greater,
                            String::from(">"),
                            Some(Value::Null),
                        ))
                    }
                }
                Some('<') => {
                    if Some('=') == self.peek() {
                        self.index += 1;
                        self.pos += 1;
                        Some(Token::new(
                            TokenType::LessEqual,
                            String::from("<="),
                            Some(Value::Null),
                        ))
                    } else {
                        Some(Token::new(
                            TokenType::Less,
                            String::from("<"),
                            Some(Value::Null),
                        ))
                    }
                }
                Some('/') => {
                    if Some('/') == self.peek() {
                        self.index += 1;
                        self.pos += 1;
                        self.take_until('\n');
                        None
                    } else {
                        Some(Token::new(
                            TokenType::Slash,
                            String::from("/"),
                            Some(Value::Null),
                        ))
                    }
                }
                Some('"') => {
                    if let Some(val) = self.collect_until('"') {
                        let inner = val.clone()[1..val.len() - 1].to_string();
                        Some(Token::new(TokenType::Str, val, Some(Value::Str(inner))))
                    } else {
                        self.report_error(LexerError::UnterminatedString(self.line, self.pos));
                        None
                    }
                }
                Some(ch) => {
                    if ch.is_numeric() {
                        let number = self.take_number();
                        Some(Token::new(
                            TokenType::Number,
                            number.clone(),
                            Some(Value::Float(number.parse().unwrap())),
                        ))
                    } else if ch.is_alphabetic() || ch == '_' {
                        let ident = self.take_identifier();
                        let token_type = match ident.as_str() {
                            "and" => TokenType::And,
                            "class" => TokenType::Class,
                            "else" => TokenType::Else,
                            "false" => TokenType::False,
                            "true" => TokenType::True,
                            "for" => TokenType::For,
                            "if" => TokenType::If,
                            "nil" => TokenType::Nil,
                            "or" => TokenType::Or,
                            "print" => TokenType::Print,
                            "return" => TokenType::Return,
                            "super" => TokenType::Super,
                            "this" => TokenType::This,
                            "var" => TokenType::Var,
                            "while" => TokenType::While,
                            "fun" => TokenType::Fun,
                            _ => TokenType::Identifier,
                        };
                        Some(Token::new(token_type, ident, Some(Value::Null)))
                    } else {
                        self.report_error(LexerError::UnexpectedToken(self.line, self.pos, ch));
                        None
                    }
                }
            };

            if let Some(token) = token {
                self.tokens.push(token);
            }

            self.index += 1;
            self.pos += 1;
        }

        if !self.tokens.last().is_some_and(|x| *x == TokenType::Eof) {
            self.tokens.push(Token::new(
                TokenType::Eof,
                String::from(""),
                Some(Value::Null),
            ));
        }

        Ok(())
    }

    pub fn dump(&self) {
        for token in self.tokens.clone().into_iter() {
            token.log();
        }
    }
}

//// PARSER
///

#[derive(Debug, Clone)]
enum Expr {
    Binary(Box<Expr>, Token, Box<Expr>),
    Literal(Value),
    Unary(Token, Box<Expr>),
    Group(Vec<Expr>),
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Unary(op, expr) => f.write_fmt(format_args!("({} {expr})", op.loxme)),
            Expr::Literal(value) => f.write_fmt(format_args!("{}", value)),
            Expr::Binary(left, op, right) => {
                f.write_fmt(format_args!("({} {left} {right})", op.loxme))
            }
            Expr::Group(tokens) => f.write_fmt(format_args!(
                "(group {})",
                tokens
                    .iter()
                    .map(|x| format!("{x}"))
                    .collect::<Vec<String>>()
                    .join(" ")
            )),
        }
    }
}

enum ParserError {
    MissingToken(TokenType),
}

// impl Debug for Expr {}

#[derive(Debug)]
struct Parser {
    tokens: Vec<Token>,
    tokens_index: usize,
    exprs: Vec<Expr>,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            tokens_index: 0,
            exprs: Vec::new(),
        }
    }

    fn next(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.tokens_index);
        self.tokens_index += 1;

        if token.is_some() {
            return Some(token.unwrap().clone());
        }

        return None;
    }

    pub fn take_until(
        &mut self,
        start: TokenType,
        end: TokenType,
    ) -> Result<Vec<Token>, ParserError> {
        let mut output = Vec::<Token>::new();
        let mut found = false;

        let mut skip = 0;

        while let Some(token) = self.next() {
            if token.token_type == start {
                skip += 1;
            }

            if token.token_type == end {
                if skip == 0 {
                    found = true;
                    break;
                } else {
                    skip -= 1;
                }
            }

            output.push(token);
        }

        if !found {
            Err(ParserError::MissingToken(end))
        } else {
            Ok(output)
        }
    }

    pub fn parse_one(&mut self, depth: usize) -> Result<Option<Expr>, ParserError> {
        let token = self.next().clone();
        if token.is_none() {
            return Ok(None);
        }

        let token = token.unwrap();

        let expr = match token.token_type {
            TokenType::Number => Some(Expr::Literal(token.value.unwrap().to_owned())),
            TokenType::True => Some(Expr::Literal(Value::Bool(true))),
            TokenType::False => Some(Expr::Literal(Value::Bool(false))),
            TokenType::Nil => Some(Expr::Literal(Value::Nil)),
            TokenType::Str => Some(Expr::Literal(token.value.unwrap().to_owned())),
            TokenType::LeftParen => {
                let tokens = self.take_until(TokenType::LeftParen, TokenType::RightParen)?;
                let expr = Expr::Group(Parser::new(tokens).parse()?);

                Some(expr)
            }
            TokenType::Bang => {
                let expr = self.parse_one(depth + 1)?.expect("expr wanted");
                Some(Expr::Unary(token, Box::new(expr)))
            }
            TokenType::Minus => {
                if depth > 0 || self.exprs.is_empty() {
                    Some(Expr::Unary(
                        token,
                        Box::new(self.parse_one(depth + 1)?.expect("Minus!!")),
                    ))
                } else {
                    let lhs = self.exprs.pop().expect("lhs expr");
                    let rhs = self.parse_one(depth + 1)?.expect("rhs expr");
                    println!("lhs: {}, op: {:?}, rhs: {}", lhs, token.token_type, rhs);
                    Some(Expr::Binary(Box::new(lhs), token, Box::new(rhs)))
                }
            }
            TokenType::Star | TokenType::Slash | TokenType::Plus => {
                let lhs = self.exprs.pop().expect("lhs expr");
                let rhs = Box::new(self.parse_one(depth + 1)?.expect("rhs expr"));

                let op = match lhs.clone() {
                    Expr::Binary(_l, op, _r) => {
                        if op == TokenType::Plus || op == TokenType::Minus {
                            Some(Expr::Binary(
                                _l,
                                op,
                                Box::new(Expr::Binary(_r, token.clone(), rhs.clone())),
                            ))
                        } else {
                            None
                        }
                    }
                    _ => None,
                };

                if op.is_some() {
                    return Ok(op);
                }

                Some(Expr::Binary(Box::new(lhs), token, rhs))
            }
            TokenType::Eof => return Ok(None),
            _ => todo!(),
        };

        Ok(expr)
    }

    pub fn parse(&mut self) -> Result<Vec<Expr>, ParserError> {
        while let Some(expr) = self.parse_one(0)? {
            self.exprs.push(expr);
        }

        Ok(self.exprs.clone())
    }

    pub fn dump(&self) {
        for expr in self.exprs.iter() {
            println!("{}", expr)
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        writeln!(io::stderr(), "Usage: {} tokenize <filename>", args[0]).unwrap();
        return Ok(());
    }

    let command = &args[1];
    let filename = &args[2];

    match command.as_str() {
        "tokenize" => {
            // You can use print statements as follows for debugging, they'll be visible when running tests.
            writeln!(io::stderr(), "Logs from your program will appear here!").unwrap();

            let file_contents = fs::read_to_string(filename).unwrap_or_else(|_| {
                writeln!(io::stderr(), "Failed to read file {}", filename).unwrap();
                String::new()
            });

            let mut lexer = Lexer::new(file_contents);

            let _ = lexer.tokenize();
            lexer.dump();

            if lexer.has_error() {
                std::process::exit(65);
            }
        }
        "parse" => {
            let file_contents = fs::read_to_string(filename).unwrap_or_else(|_| {
                writeln!(io::stderr(), "Failed to read file {}", filename).unwrap();
                String::new()
            });

            let mut lexer = Lexer::new(file_contents);
            let _ = lexer.tokenize();
            if lexer.has_error() {
                std::process::exit(65);
            }
            let mut parser = Parser::new(lexer.tokens);

            if parser.parse().is_ok() {
                parser.dump();
            } else {
                println!("Error!");
            }
        }
        _ => {
            writeln!(io::stderr(), "Unknown command: {}", command).unwrap();
        }
    }

    Ok(())
}
