use crate::Result;
use candle_core::Tensor;
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PoolingStrategy {
    Mean,
    Max,
    Sum,
}

pub fn pool_embeddings(embeddings: &Tensor, strategy: &PoolingStrategy) -> Result<Tensor> {
    match strategy {
        PoolingStrategy::Mean => mean_pooling(embeddings),
        PoolingStrategy::Max => max_pooling(embeddings),
        PoolingStrategy::Sum => sum_pooling(embeddings),
    }
}

pub fn mean_pooling(embeddings: &Tensor) -> Result<Tensor> {
    let (_, out_tokens, _) = embeddings.dims3()?;

    Ok((embeddings.sum(1)? / (out_tokens as f64))?)
}

pub fn max_pooling(embeddings: &Tensor) -> Result<Tensor> {
    Ok(embeddings.max(1)?)
}

pub fn sum_pooling(embeddings: &Tensor) -> Result<Tensor> {
    Ok(embeddings.sum(1)?)
}

#[cfg(test)]
mod test {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn test_pooling_strategy_helper(
        strategy: &PoolingStrategy,
        assert_op: fn(&Vec<Vec<f32>>) -> bool,
    ) -> Result<()> {
        // 1 sentence, 20 tokens, 32 dimensions
        let v = Tensor::ones(&[1, 20, 32], DType::F32, &Device::Cpu)?;
        let v_pool = pool_embeddings(&v, strategy)?;
        let (sent, dim) = v_pool.dims2()?;
        assert_eq!(sent, 1);
        assert_eq!(dim, 32);

        let v_vec = v_pool.to_vec2::<f32>()?;
        assert!(assert_op(&v_vec));

        Ok(())
    }

    #[test]
    fn test_mean_pooling() -> Result<()> {
        test_pooling_strategy_helper(&PoolingStrategy::Mean, |v_vec| {
            v_vec[0].iter().all(|&x| x == 1.0)
        })?;

        Ok(())
    }

    #[test]
    fn test_max_pooling() -> Result<()> {
        test_pooling_strategy_helper(&PoolingStrategy::Max, |v_vec| {
            v_vec[0].iter().all(|&x| x == 1.0)
        })?;

        Ok(())
    }

    #[test]
    fn test_sum_pooling() -> Result<()> {
        test_pooling_strategy_helper(&PoolingStrategy::Sum, |v_vec| {
            v_vec[0].iter().all(|&x| x == 20.0)
        })?;

        Ok(())
    }
}
