import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/raw_points.csv")

# Make sure Points is numeric
df["Points"] = pd.to_numeric(df["Points"], errors="coerce").fillna(0)

# ---------- Longest Streak ----------
streak_df = df[["Name", "Date"]].copy()
streak_df["Date"] = pd.to_datetime(streak_df["Date"], dayfirst=True)

# Count only one award per student per day
streak_df = streak_df.drop_duplicates()

streak_df = streak_df.sort_values(["Name", "Date"])

prev_date = streak_df["Date"].shift()

is_consecutive = (
    (streak_df["Date"] == prev_date + pd.Timedelta(days=1)) |
    (
        (prev_date.dt.weekday == 4) &
        (streak_df["Date"] == prev_date + pd.Timedelta(days=3))
    )
)

streak_df["streak_break"] = (
    (streak_df["Name"] != streak_df["Name"].shift()) |
    (~is_consecutive)
)

streak_df["streak_id"] = streak_df["streak_break"].cumsum()

streaks = (
    streak_df
    .groupby(["Name", "streak_id"])
    .size()
    .reset_index(name="streak_length")
)

longest = streaks.sort_values("streak_length", ascending=False).iloc[0]

print(f"Longest streak: {longest['Name']} with {longest['streak_length']} days")

streak_leaderboard = (
    streaks
    .groupby("Name")["streak_length"]
    .max()
    .reset_index()
    .sort_values("streak_length", ascending=False)
)

# ---------- Top Teacher ----------
teacher_points = (
    df.groupby("Teacher", dropna=False)["Points"]
      .sum()
      .reset_index()
      .sort_values("Points", ascending=False)
)

top_teacher = teacher_points.iloc[0]

print(
    f"Top teacher: {top_teacher['Teacher']} "
    f"with {top_teacher['Points']} total points given"
)

# ---------- Top Form ----------
form_points = (
    df.groupby("Form", dropna=False)["Points"]
      .sum()
      .reset_index()
      .sort_values("Points", ascending=False)
)

top_form = form_points.iloc[0]

print(
    f"Top form: {top_form['Form']} "
    f"with {top_form['Points']} total points given"
)

# ---------- Top Student ----------
student_points = (
    df.groupby("Name")["Points"]
      .sum()
      .reset_index()
      .sort_values("Points", ascending=False)
)

top_student = student_points.iloc[0]

print(
    f"Top student: {top_student['Name']} "
    f"with {top_student['Points']} total points given"
)

# ---------- Teacher Dependence ----------
student_teacher_points = (
    df.groupby(["Name", "Teacher"])["Points"]
      .sum()
      .reset_index()
)

student_totals = (
    df.groupby("Name")["Points"]
      .sum()
      .reset_index(name="total_points")
)

dependence = student_teacher_points.merge(
    student_totals,
    on="Name"
)

dependence["dependence_pct"] = (
    dependence["Points"] / dependence["total_points"] * 100
)

# For each student, keep the teacher they depend on most
max_dependence = (
    dependence
    .sort_values("dependence_pct", ascending=False)
    .groupby("Name")
    .first()
    .reset_index()
)

# Sort overall leaderboard
dependence_leaderboard = max_dependence.sort_values(
    "dependence_pct",
    ascending=False
)

# ---------- Longest No House Point Streak (including today) ----------
no_point_df = df[["Name", "Date"]].copy()
no_point_df["Date"] = pd.to_datetime(no_point_df["Date"], dayfirst=True)

# One award per student per day
no_point_df = no_point_df.drop_duplicates()

# Sort
no_point_df = no_point_df.sort_values(["Name", "Date"])

TODAY = pd.Timestamp.today().normalize()

def school_days_between(d1, d2):
    days = pd.date_range(d1 + pd.Timedelta(days=1), d2 - pd.Timedelta(days=1))
    return sum(day.weekday() < 5 for day in days)

results = []

for name, group in no_point_df.groupby("Name"):
    dates = group["Date"].tolist()

    # Pretend they got a point today
    if dates[-1] < TODAY:
        dates.append(TODAY)

    max_gap = 0

    for prev, curr in zip(dates, dates[1:]):
        gap = school_days_between(prev, curr)
        max_gap = max(max_gap, gap)

    results.append({
        "Name": name,
        "longest_no_house_point_streak": max_gap
    })

no_point_streaks = (
    pd.DataFrame(results)
      .sort_values("longest_no_house_point_streak", ascending=False)
)

# ---------- Gini Coefficient ----------
import numpy as np

student_totals = df.groupby("Name")["Points"].sum().values

def gini(x):
    x = np.array(x)
    if np.all(x == 0):
        return 0
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * x) / (n * np.sum(x))) - (n + 1) / n

gini_score = gini(student_totals)
print(f"Gini coefficient (student points inequality): {round(gini_score, 3)}")

# ---------- Current Drought ----------
today = pd.Timestamp.today().normalize()

droughts = []

for name, group in no_point_df.groupby("Name"):
    last_date = group["Date"].max()
    drought = school_days_between(last_date, today)

    droughts.append({
        "Name": name,
        "current_drought_days": drought
    })

current_droughts = (
    pd.DataFrame(droughts)
      .sort_values("current_drought_days", ascending=False)
)

# ---------- Teacher Bias Score ----------
teacher_bias = []

for teacher, group in df.groupby("Teacher"):
    totals = (
        group.groupby("Name")["Points"]
        .sum()
        .sort_values(ascending=False)
    )

    top_5_share = totals.head(1).sum() / totals.sum() * 100

    teacher_bias.append({
        "Teacher": teacher,
        "top_5_student_share_pct": round(top_5_share, 2)
    })

teacher_bias_df = (
    pd.DataFrame(teacher_bias)
      .sort_values("top_5_student_share_pct", ascending=False)
)

# ---------- Consistency Score ----------
consistency = []

for name, group in no_point_df.groupby("Name"):
    dates = group["Date"].tolist()
    gaps = [
        school_days_between(d1, d2)
        for d1, d2 in zip(dates, dates[1:])
    ]

    if len(gaps) > 1:
        consistency.append({
            "Name": name,
            "consistency_score": round(np.std(gaps), 2)
        })

consistency_df = (
    pd.DataFrame(consistency)
      .sort_values("consistency_score")
)

# ---------- Exclusive Pairs ----------
exclusive_pairs = []

student_teacher = (
    df.groupby(["Name", "Teacher"])["Points"]
      .sum()
      .reset_index()
)

student_totals = (
    df.groupby("Name")["Points"]
      .sum()
      .reset_index(name="student_total")
)

teacher_totals = (
    df.groupby("Teacher")["Points"]
      .sum()
      .reset_index(name="teacher_total")
)

merged = (
    student_teacher
    .merge(student_totals, on="Name")
    .merge(teacher_totals, on="Teacher")
)

merged["student_dependence_pct"] = merged["Points"] / merged["student_total"] * 100
merged["teacher_focus_pct"] = merged["Points"] / merged["teacher_total"] * 100

exclusive_pairs = merged[
    (merged["student_dependence_pct"] >= 30) &
    (merged["teacher_focus_pct"] >= 30)
]

# ---------- Save rankings ----------
streak_leaderboard.to_csv("tables/streak_leaderboard.csv", index=False)
teacher_points.to_csv("tables/teacher_points.csv", index=False)
form_points.to_csv("tables/form_points.csv", index=False)
student_points.to_csv("tables/student_points.csv", index=False)
dependence_leaderboard.to_csv("tables/teacher_dependence_leaderboard.csv", index=False)
no_point_streaks.to_csv("tables/no_house_point_streak_leaderboard.csv", index=False)
current_droughts.to_csv("tables/current_drought_leaderboard.csv", index=False)
teacher_bias_df.to_csv("tables/teacher_bias_score.csv", index=False)
consistency_df.to_csv("tables/consistency_score.csv", index=False)
exclusive_pairs.to_csv("tables/exclusive_pairs.csv", index=False)

top_students_plot = student_points.head(10)

plt.figure()
plt.barh(
    top_students_plot["Name"][::-1],
    top_students_plot["Points"][::-1]
)
plt.title("Top 10 Students by Total Points")
plt.xlabel("Total Points")
plt.ylabel("Student")
plt.tight_layout()
plt.savefig("graphs/top_students.png")
plt.close()

top_streaks_plot = streak_leaderboard.head(10)

plt.figure()
plt.barh(
    top_streaks_plot["Name"][::-1],
    top_streaks_plot["streak_length"][::-1]
)
plt.title("Top 10 Longest House Point Streaks")
plt.xlabel("Streak Length (Days)")
plt.ylabel("Student")
plt.tight_layout()
plt.savefig("graphs/longest_streaks.png")
plt.close()

top_droughts_plot = no_point_streaks.head(10)

plt.figure()
plt.barh(
    top_droughts_plot["Name"][::-1],
    top_droughts_plot["longest_no_house_point_streak"][::-1]
)
plt.title("Top 10 Longest No House Point Streaks")
plt.xlabel("School Days Without Points")
plt.ylabel("Student")
plt.tight_layout()
plt.savefig("graphs/longest_droughts.png")
plt.close()

top_teacher_bias_plot = teacher_bias_df.head(10)

plt.figure()
plt.barh(
    top_teacher_bias_plot["Teacher"][::-1],
    top_teacher_bias_plot["top_5_student_share_pct"][::-1]
)
plt.title("Teacher Bias Score (Top 5 Student Share %)")
plt.xlabel("Percentage of Points")
plt.ylabel("Teacher")
plt.tight_layout()
plt.savefig("graphs/teacher_bias.png")
plt.close()

top_dependence_plot = dependence_leaderboard.head(10)

plt.figure()
plt.barh(
    top_dependence_plot["Name"][::-1],
    top_dependence_plot["dependence_pct"][::-1]
)
plt.title("Highest Teacher Dependence (Top 10 Students)")
plt.xlabel("Percentage of Points from One Teacher")
plt.ylabel("Student")
plt.tight_layout()
plt.savefig("graphs/teacher_dependence.png")
plt.close()

plt.figure()
plt.hist(consistency_df["consistency_score"], bins=20)
plt.title("Consistency Score Distribution")
plt.xlabel("Consistency Score (Std Dev of Gaps)")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("graphs/consistency_distribution.png")
plt.close()

plt.figure()
plt.hist(student_points["Points"], bins=50)
plt.title("Distribution of Total Points per Student")
plt.xlabel("Total Points")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("graphs/points_distribution.png")
plt.close()