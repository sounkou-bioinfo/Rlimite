args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0L) {
  stop("usage: normalize-markdown.R path [path ...]", call. = FALSE)
}

for (path in args) {
  lines <- readLines(path, warn = FALSE)
  lines <- sub("[ \t]+$", "", lines, perl = TRUE)
  writeLines(lines, path, useBytes = TRUE)
}
