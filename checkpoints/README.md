## Split large file into 100MiB parts

```bash
split -b 100MiB 10.pth 10.pth.part.
```

## Recombine parts into original file

```bash
cat 10.pth.part.* > 10.pth
```
