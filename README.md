- for submission

```
g++ KKT89/_Combine.cpp -O3 -Ofast -std=c++17 -o main \
    -mmmx -msse -msse2 -msse3 -mssse3 -mmovbe -maes -mpclmul -mpopcnt -mfma -mbmi -mbmi2 \
    -mavx -mavx2 -msse4.2 -msse4.1 -mlzcnt -mrdrnd -mf16c -mfsgsbase
```

- for self-play

```
g++ KKT89/Combine.cpp -O3 -Ofast -std=c++17 -o main -march=native
```
