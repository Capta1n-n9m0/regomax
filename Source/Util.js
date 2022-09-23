const Vector = require("./Vector");

/**
 * @return {number}
 */
function getTime() {
    const delay = Number(process.hrtime.bigint() / BigInt(1_000));
    return delay / 1_000;
}

/**
 * @param {Vector} right
 * @param {Vector} left
 * @param {Vector} v
 */
function projectQ(right, left, v) {
    let sp;

    sp = Vector.scalar_product(left, v);
    v.lam_diff(sp, right);
}

/**
 * @param {Vector} right
 * @param {Vector} left
 * @param {Vector} v
 * @param {number} f
 */
function projectP(right, left, v, f = 1) {
    let i, n = v.size();
    let sp;

    sp = Vector.scalar_product(left, v) / f;
    v.test(right);
    for (i = 0; i < n; i++) v.c[i] = sp * right.c[i];
}

/**
 * @param {Vector} a
 * @param {Vector} b
 * @return number
 */
function diff_norm_rel(a, b) {
    let sum, ss;
    let i, n;
    n = a.size();
    sum = 0.0;
    //#pragma omp parallel for reduction(+:sum)
    for (i = 0; i < n; i++) {
        ss = Vector.abs(a.c[i]) + Vector.abs(b.c[i]);
        if (ss === 0) continue;
        sum += Vector.abs(a.c[i] - b.c[i]) / ss;
    }
    return sum;
}

/**
 * @param {Vector} a
 * @return number
 */
function pagerank_normalize(a) {
    let sum;

    sum = Vector.sum_vector(a);
    a.div_eq(sum);
    return sum;
}




module.exports = {getTime, projectQ, projectP, diff_norm_rel, pagerank_normalize};