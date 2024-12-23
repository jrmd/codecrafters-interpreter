use core::fmt;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::error::Error;

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
enum LexerError {
    ParseError(usize, usize),
    UnexpectedToken(usize, usize, char),
}

impl fmt::Display for LexerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LexerError::ParseError(line, _) => write!(f, "[line {}] Error: Parse Error", line),
            LexerError::UnexpectedToken(line, _, ch) => write!(f, "[line {}] Error: Unexpected character: {}", line, ch),
        }
    }
}

impl std::error::Error for LexerError  {
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
    errors: Vec<LexerError>
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

    pub fn parse(&mut self) -> Result<(), LexerError> {
        let strlen = self.input.len();

        while self.index < strlen {
            self.take_whitespace();
            let token = match self.input.chars().nth(self.index) {
                None => {
                    if (self.tokens.last().is_some_and(|x| *x == TokenType::Eof)) {
                        None
                    } else {
                        Some(Token::new(TokenType::Eof, String::from(""), Some(Value::Null)))
                    }
                }
                Some('(') => Some(Token::new(TokenType::LeftParen, String::from("("), Some(Value::Null))),
                Some(')') => Some(Token::new(TokenType::RightParen, String::from(")"), Some(Value::Null))),
                Some('{') => Some(Token::new(TokenType::LeftBrace, String::from("{"), Some(Value::Null))),
                Some('}') => Some(Token::new(TokenType::RightBrace, String::from("}"), Some(Value::Null))),
                Some('*') => Some(Token::new(TokenType::Star, String::from("*"), Some(Value::Null))),
                Some('.') => Some(Token::new(TokenType::Period, String::from("."), Some(Value::Null))),
                Some(',') => Some(Token::new(TokenType::Comma, String::from(","), Some(Value::Null))),
                Some('+') => Some(Token::new(TokenType::Plus, String::from("+"), Some(Value::Null))),
                Some('-') => Some(Token::new(TokenType::Minus, String::from("-"), Some(Value::Null))),
                Some(';') => Some(Token::new(TokenType::Semicolon, String::from(";"), Some(Value::Null))),
                Some('=') => {
                    if Some('=') == self.peek() {
                        self.index += 1;
                        self.pos += 1;
                        Some(Token::new(TokenType::EqualEqual, String::from("=="), Some(Value::Null)))
                    } else {
                        Some(Token::new(TokenType::Equal, String::from("="), Some(Value::Null)))
                    }
                },
                Some('!') => {
                    if Some('=') == self.peek() {
                        self.index += 1;
                        self.pos += 1;
                        Some(Token::new(TokenType::BangEqual, String::from("!="), Some(Value::Null)))
                    } else {
                        Some(Token::new(TokenType::Bang, String::from("!"), Some(Value::Null)))
                    }
                }
                Some('>') => {
                    if Some('=') == self.peek() {
                        self.index += 1;
                        self.pos += 1;
                        Some(Token::new(TokenType::GreaterEqual, String::from(">="), Some(Value::Null)))
                    } else {
                        Some(Token::new(TokenType::Greater, String::from(">"), Some(Value::Null)))
                    }
                },
                Some('<') => {
                    if Some('=') == self.peek() {
                        self.index += 1;
                        self.pos += 1;
                        Some(Token::new(TokenType::LessEqual, String::from("<="), Some(Value::Null)))
                    } else {
                        Some(Token::new(TokenType::Less, String::from("<"), Some(Value::Null)))
                    }
                },
                Some('/') => {
                    if Some('/') == self.peek() {
                        self.index += 1;
                        self.pos += 1;
                        while let Some(ch) = self.peek() {
                            if ch != '\n' {
                                self.index += 1;
                                self.pos += 1;
                                continue;
                            }

                            break;
                        }
                        None
                    } else {
                        Some(Token::new(TokenType::Slash, String::from("/"), Some(Value::Null)))
                    }
                }
                Some(ch) => {
                    self.report_error(LexerError::UnexpectedToken(self.line, self.pos, ch));
                    None
                },
            };

            if let Some(token) = token {
                self.tokens.push(token);
            }

            self.index += 1;
            self.pos += 1;
        }

        if !self.tokens.last().is_some_and(|x| *x == TokenType::Eof) {
            self.tokens.push(Token::new(TokenType::Eof, String::from(""), Some(Value::Null)));
        }

        Ok(())
    }
 
    pub fn dump(&self) {
        for token in self.tokens.clone().into_iter() {
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

            let _ = lexer.parse();
            lexer.dump();

            if lexer.has_error() {
                std::process::exit(65);
            }
        }
        _ => {
            writeln!(io::stderr(), "Unknown command: {}", command).unwrap();
        }
    }

    Ok(())
}
