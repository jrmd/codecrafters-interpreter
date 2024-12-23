use core::fmt;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::error::Error;

#[derive(Debug)]
enum Value {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Null,
}
impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Str(val) => write!(f, "\"{}\"", val),
            Value::Int(val) => write!(f, "\"{}\"", val),
            Value::Float(val) => write!(f, "\"{}\"", val),
            Value::Bool(val) => write!(f, "{}", if *val { "true" } else { "false" }),
            Value::Null => write!(f, "{}", String::from("null")),
        }
    }
}

#[derive(Debug, PartialEq)]
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

    And, Class, Else, False, Fun, For, If, Nil, Or,
    Print, Return, Super, This, True, Var, While,

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
            TokenType::Period => "PERIOD",
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
            TokenType::Str => "STR",
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

#[derive(Debug)]
enum LexerError {
    ParseError,
    UnexpectedToken,
}

impl fmt::Display for LexerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Lexer Error")
    }
}

impl std::error::Error for LexerError  {
    fn description(&self) -> &str {
        "lexer error"
    }
}

#[derive(Debug)]
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
}

impl Lexer {
    pub fn new(input: String) -> Self {
        Self {
            tokens: Vec::new(),
            input,
            index: 0,
        }
    }

    fn take_whitespace(&mut self) {
        while let Some(ch) = self.input.chars().nth(self.index) {
            if ch.is_whitespace() {
                self.index += 1;
            } else {
                break;
            }
        }
    }

    pub fn parse(&mut self) -> Result<(), LexerError> {
        let strlen = self.input.len();
        
        while self.index < strlen {
            self.take_whitespace();

            let token = match self.input.chars().nth(self.index) {
                Some('(') => Some(Token::new(TokenType::LeftParen, String::from("("), Some(Value::Null))),
                Some(')') => Some(Token::new(TokenType::RightParen, String::from(")"), Some(Value::Null))),
                Some('{') => Some(Token::new(TokenType::LeftBrace, String::from("{"), Some(Value::Null))),
                Some('}') => Some(Token::new(TokenType::RightBrace, String::from("}"), Some(Value::Null))),
                None => Some(Token::new(TokenType::Eof, String::from(""), Some(Value::Null))),
                _ => None,
            };

            if let Some(token) = token {
                self.tokens.push(token);
            } else {
                println!("--{}--", self.input.chars().nth(self.index).unwrap());
                return Err(LexerError::UnexpectedToken)
            }

            self.index += 1;
        }

        if !self.tokens.last().is_some_and(|x| *x == TokenType::Eof) {
            self.tokens.push(Token::new(TokenType::Eof, String::from(""), Some(Value::Null)));
        }

        Ok(())
    }

    pub fn dump(self) {
        for token in self.tokens {
           token.log();
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

            lexer.parse()?;

            lexer.dump();
        }
        _ => {
            writeln!(io::stderr(), "Unknown command: {}", command).unwrap();
        }
    }

    Ok(())
}
