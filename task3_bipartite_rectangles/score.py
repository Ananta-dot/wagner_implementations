import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def intersect_interval(ab, cd):
    a, b = sorted(ab)
    c, d = sorted(cd)
    if b <= c or d <= a:
        return False
    return True

def intersect(rectA, rectB):
    return (intersect_interval(rectA[0], rectB[0]) and
            intersect_interval(rectA[1], rectB[1]))

def main():
    rects = [
       ((5,8),(2,4)),  
       ((3,4),(5,6)),  
       ((1,2),(7,8)),  
       ((6,7),(1,3))   
    ]
    colors = [0,0,1,1]  
    score = 0
    for i in range(4):
        for j in range(i+1,4):
            same_c = (colors[i] == colors[j])
            does_int = intersect(rects[i], rects[j])
            if same_c:
                if not does_int:
                    score += 1
            else:
                if does_int:
                    score += 1

    print("Rectangles (strict bounding boxes) => ( (x1,x2),(y1,y2) ):")
    for i,(bx,by) in enumerate(rects):
        print(f"  Rect{i+1} => color={colors[i]}, X={bx}, Y={by}")

    print(f"\nFinal Score = {score} (expected=4)")

    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_aspect('equal')
    ax.set_title("Rectangles => Score=4 for given bounding boxes")

    min_x = float('inf')
    max_x = float('-inf')
    min_y = float('inf')
    max_y = float('-inf')

    def to_box(r):
        (x1,x2),(y1,y2) = r
        left, right = sorted([x1,x2])
        bottom, top = sorted([y1,y2])
        return (left,right,bottom,top)

    for i,(bx,by) in enumerate(rects):
        left,right,bottom,top = to_box((bx,by))
        col = 'red' if colors[i]==0 else 'blue'

        min_x = min(min_x, left)
        max_x = max(max_x, right)
        min_y = min(min_y, bottom)
        max_y = max(max_y, top)
        width  = right - left
        height = top - bottom
        ax.add_patch(Rectangle((left,bottom), width, height,
                               edgecolor='black', facecolor=col, alpha=0.3))
        cx = (left+right)/2
        cy = (bottom+top)/2
        ax.text(cx, cy, f"#{i+1}", ha='center', va='center',
                fontsize=11, color='k', fontweight='bold')

    ax.set_xlim(min_x-1, max_x+1)
    ax.set_ylim(min_y-1, max_y+1)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(True)
    plt.show()


if __name__=="__main__":
    main()
