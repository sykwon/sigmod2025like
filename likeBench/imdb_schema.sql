CREATE TABLE aka_name (
    id integer NOT NULL,
    person_id integer NOT NULL,
    name varchar(255),
    imdb_index varchar(3),
    name_pcode_cf varchar(11),
    name_pcode_nf varchar(11),
    surname_pcode varchar(11),
    md5sum varchar(65),
    PRIMARY KEY(id)
);

CREATE TABLE aka_title (
    id integer NOT NULL,
    movie_id integer NOT NULL,
    title varchar(600),
    imdb_index varchar(4),
    kind_id integer NOT NULL,
    production_year integer,
    phonetic_code varchar(5),
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    note varchar(72),
    md5sum varchar(33),
    PRIMARY KEY(id)
);

CREATE TABLE cast_info (
    id integer NOT NULL,
    person_id integer NOT NULL,
    movie_id integer NOT NULL,
    person_role_id integer,
    note varchar(1000),
    nr_order integer,
    role_id integer NOT NULL,
    PRIMARY KEY(id)
);

CREATE TABLE char_name (
    id integer NOT NULL,
    name varchar(500) NOT NULL,
    imdb_index varchar(2),
    imdb_id integer,
    name_pcode_nf varchar(5),
    surname_pcode varchar(5),
    md5sum varchar(33),
    PRIMARY KEY(id)
);

CREATE TABLE comp_cast_type (
    id integer NOT NULL,
    kind varchar(32) NOT NULL,
    PRIMARY KEY(id)
);

CREATE TABLE company_name (
    id integer NOT NULL,
    name varchar(255) NOT NULL,
    country_code varchar(6),
    imdb_id integer,
    name_pcode_nf varchar(5),
    name_pcode_sf varchar(5),
    md5sum varchar(33),
    PRIMARY KEY(id)
);

CREATE TABLE company_type (
    id integer NOT NULL,
    kind varchar(32),
    PRIMARY KEY(id)
);

CREATE TABLE complete_cast (
    id integer NOT NULL,
    movie_id integer,
    subject_id integer NOT NULL,
    status_id integer NOT NULL,
    PRIMARY KEY(id)
);

CREATE TABLE info_type (
    id integer NOT NULL,
    info varchar(32) NOT NULL,
    PRIMARY KEY(id)
);

CREATE TABLE keyword (
    id integer NOT NULL,
    keyword varchar(100) NOT NULL,
    phonetic_code varchar(6),
    PRIMARY KEY(id)
);

CREATE TABLE kind_type (
    id integer NOT NULL,
    kind varchar(15),
    PRIMARY KEY(id)
);

CREATE TABLE link_type (
    id integer NOT NULL,
    link varchar(32) NOT NULL,
    PRIMARY KEY(id)
);

CREATE TABLE movie_companies (
    id integer NOT NULL,
    movie_id integer NOT NULL,
    company_id integer NOT NULL,
    company_type_id integer NOT NULL,
    note varchar(255),
    PRIMARY KEY(id)
);

CREATE TABLE movie_info_idx (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info varchar(10) NOT NULL,
    note varchar(1)
);

CREATE TABLE movie_keyword (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    keyword_id integer NOT NULL
);

CREATE TABLE movie_link (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    linked_movie_id integer NOT NULL,
    link_type_id integer NOT NULL
);

CREATE TABLE name (
    id integer NOT NULL PRIMARY KEY,
    name varchar(125) NOT NULL,
    imdb_index varchar(9),
    imdb_id integer,
    gender varchar(1),
    name_pcode_cf varchar(5),
    name_pcode_nf varchar(5),
    surname_pcode varchar(5),
    md5sum varchar(33)
);

CREATE TABLE role_type (
    id integer NOT NULL PRIMARY KEY,
    role varchar(32) NOT NULL
);

CREATE TABLE title (
    id integer NOT NULL PRIMARY KEY,
    title varchar(350) NOT NULL,
    imdb_index varchar(5),
    kind_id integer NOT NULL,
    production_year integer,
    imdb_id integer,
    phonetic_code varchar(5),
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    series_years varchar(49),
    md5sum varchar(33)
);

CREATE TABLE movie_info (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info text NOT NULL,
    note varchar(500)
);

CREATE TABLE person_info (
    id integer NOT NULL PRIMARY KEY,
    person_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info text NOT NULL,
    note varchar(500)
);

