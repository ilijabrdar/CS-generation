cd %1
for /d %%f in (*) do (
    cd %cd%\%%f
    rename *.txt *.java
    mkdir %2\%%f
    cd %2\%%f
    java -jar %4 %1\%%f false 0 False
    cd %1
)
