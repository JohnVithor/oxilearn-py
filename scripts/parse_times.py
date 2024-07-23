import sys
import re
import csv


def main(input_file_name, output_file_name):
    padrao = re.compile(r"(\d+m\d+,\d+s)")
    tempos = []
    with open(input_file_name, "r") as arquivo:
        for tempo in padrao.findall(arquivo.read()):
            min, secs = tempo.split("m")
            min, secs = float(min), float(secs.replace(",", ".")[:-1])
            tempos.append(60 * min + secs)

    with open(f"{output_file_name}.csv", "w", newline="") as arquivo_csv:
        escritor_csv = csv.writer(arquivo_csv)
        escritor_csv.writerow(["id", "version", "real", "user", "sys"])
        for i in range(len(tempos) // 3):
            v = "rust" if i % 2 == 0 else "python"
            escritor_csv.writerow([i // 2] + [v] + tempos[i * 3 : i * 3 + 3])


if __name__ == "__main__":
    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]

    main(input_file_name, output_file_name)
