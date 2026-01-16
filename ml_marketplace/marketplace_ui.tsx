import React, { useState, useEffect } from 'react';
import {
  Container,
  Grid,
  Card,
  CardContent,
  CardActions,
  Typography,
  Button,
  TextField,
  Select,
  MenuItem,
  Chip,
  Rating,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  LinearProgress,
  Alert,
  Tabs,
  Tab,
  Box
} from '@mui/material';
import {
  CloudUpload,
  Download,
  Star,
  TrendingUp,
  Assessment,
  Science
} from '@mui/icons-material';
import axios from 'axios';

interface MLModel {
  id: string;
  name: string;
  version: string;
  description: string;
  category: string;
  author: string;
  created_at: string;
  updated_at: string;
  downloads: number;
  rating: number;
  reviews: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  file_size: string;
  framework: string;
  tags: string[];
  status: 'active' | 'deprecated' | 'experimental';
}

interface ABTest {
  id: string;
  name: string;
  model_a: string;
  model_b: string;
  status: 'running' | 'completed' | 'paused';
  start_date: string;
  end_date?: string;
  sample_size: number;
  confidence_level: number;
  p_value?: number;
  winner?: string;
}

const MLMarketplace: React.FC = () => {
  const [models, setModels] = useState<MLModel[]>([]);
  const [abTests, setABTests] = useState<ABTest[]>([]);
  const [selectedModel, setSelectedModel] = useState<MLModel | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [categoryFilter, setCategoryFilter] = useState('all');
  const [sortBy, setSortBy] = useState('downloads');
  const [tabValue, setTabValue] = useState(0);
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchModels();
    fetchABTests();
  }, []);

  const fetchModels = async () => {
    setLoading(true);
    try {
      const response = await axios.get('/api/ml-marketplace/models');
      setModels(response.data);
    } catch (error) {
      console.error('Error fetching models:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchABTests = async () => {
    try {
      const response = await axios.get('/api/ml-marketplace/ab-tests');
      setABTests(response.data);
    } catch (error) {
      console.error('Error fetching A/B tests:', error);
    }
  };

  const handleDownloadModel = async (modelId: string) => {
    try {
      const response = await axios.get(`/api/ml-marketplace/models/${modelId}/download`, {
        responseType: 'blob'
      });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `model_${modelId}.h5`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error('Error downloading model:', error);
    }
  };

  const handleUploadModel = async (formData: FormData) => {
    try {
      await axios.post('/api/ml-marketplace/models/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setUploadDialogOpen(false);
      fetchModels();
    } catch (error) {
      console.error('Error uploading model:', error);
    }
  };

  const handleCreateABTest = async (testData: any) => {
    try {
      await axios.post('/api/ml-marketplace/ab-tests', testData);
      fetchABTests();
    } catch (error) {
      console.error('Error creating A/B test:', error);
    }
  };

  const filteredModels = models
    .filter(model => 
      (categoryFilter === 'all' || model.category === categoryFilter) &&
      (searchQuery === '' || 
        model.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        model.description.toLowerCase().includes(searchQuery.toLowerCase())
      )
    )
    .sort((a, b) => {
      switch (sortBy) {
        case 'downloads':
          return b.downloads - a.downloads;
        case 'rating':
          return b.rating - a.rating;
        case 'accuracy':
          return b.accuracy - a.accuracy;
        case 'recent':
          return new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime();
        default:
          return 0;
      }
    });

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Typography variant="h3" gutterBottom>
        ML Model Marketplace
      </Typography>

      <Tabs value={tabValue} onChange={(e, v) => setTabValue(v)} sx={{ mb: 3 }}>
        <Tab label="Browse Models" icon={<Science />} />
        <Tab label="A/B Testing" icon={<Assessment />} />
        <Tab label="My Models" icon={<CloudUpload />} />
      </Tabs>

      {tabValue === 0 && (
        <>
          <Grid container spacing={2} sx={{ mb: 3 }}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Search models"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search by name or description..."
              />
            </Grid>
            <Grid item xs={12} md={3}>
              <Select
                fullWidth
                value={categoryFilter}
                onChange={(e) => setCategoryFilter(e.target.value)}
              >
                <MenuItem value="all">All Categories</MenuItem>
                <MenuItem value="classification">Classification</MenuItem>
                <MenuItem value="segmentation">Segmentation</MenuItem>
                <MenuItem value="detection">Detection</MenuItem>
                <MenuItem value="prediction">Prediction</MenuItem>
              </Select>
            </Grid>
            <Grid item xs={12} md={3}>
              <Select
                fullWidth
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
              >
                <MenuItem value="downloads">Most Downloaded</MenuItem>
                <MenuItem value="rating">Highest Rated</MenuItem>
                <MenuItem value="accuracy">Best Accuracy</MenuItem>
                <MenuItem value="recent">Most Recent</MenuItem>
              </Select>
            </Grid>
          </Grid>

          {loading && <LinearProgress />}

          <Grid container spacing={3}>
            {filteredModels.map((model) => (
              <Grid item xs={12} md={6} lg={4} key={model.id}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      {model.name}
                      <Chip 
                        label={model.status} 
                        size="small" 
                        color={model.status === 'active' ? 'success' : 'warning'}
                        sx={{ ml: 1 }}
                      />
                    </Typography>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      v{model.version} • {model.framework}
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 2 }}>
                      {model.description}
                    </Typography>
                    
                    <Box sx={{ mb: 2 }}>
                      {model.tags.map((tag) => (
                        <Chip key={tag} label={tag} size="small" sx={{ mr: 0.5, mb: 0.5 }} />
                      ))}
                    </Box>

                    <Grid container spacing={1}>
                      <Grid item xs={6}>
                        <Typography variant="caption" color="text.secondary">
                          Accuracy
                        </Typography>
                        <Typography variant="body2" fontWeight="bold">
                          {(model.accuracy * 100).toFixed(1)}%
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="caption" color="text.secondary">
                          F1 Score
                        </Typography>
                        <Typography variant="body2" fontWeight="bold">
                          {model.f1_score.toFixed(3)}
                        </Typography>
                      </Grid>
                    </Grid>

                    <Box sx={{ display: 'flex', alignItems: 'center', mt: 2 }}>
                      <Rating value={model.rating} readOnly size="small" />
                      <Typography variant="caption" sx={{ ml: 1 }}>
                        ({model.reviews} reviews)
                      </Typography>
                    </Box>

                    <Typography variant="caption" color="text.secondary">
                      {model.downloads.toLocaleString()} downloads • {model.file_size}
                    </Typography>
                  </CardContent>
                  <CardActions>
                    <Button 
                      size="small" 
                      startIcon={<Download />}
                      onClick={() => handleDownloadModel(model.id)}
                    >
                      Download
                    </Button>
                    <Button size="small" onClick={() => setSelectedModel(model)}>
                      Details
                    </Button>
                  </CardActions>
                </Card>
              </Grid>
            ))}
          </Grid>
        </>
      )}

      {tabValue === 1 && (
        <>
          <Button 
            variant="contained" 
            sx={{ mb: 3 }}
            onClick={() => {/* Open create A/B test dialog */}}
          >
            Create New A/B Test
          </Button>

          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Test Name</TableCell>
                <TableCell>Model A</TableCell>
                <TableCell>Model B</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Sample Size</TableCell>
                <TableCell>P-Value</TableCell>
                <TableCell>Winner</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {abTests.map((test) => (
                <TableRow key={test.id}>
                  <TableCell>{test.name}</TableCell>
                  <TableCell>{test.model_a}</TableCell>
                  <TableCell>{test.model_b}</TableCell>
                  <TableCell>
                    <Chip 
                      label={test.status} 
                      color={test.status === 'completed' ? 'success' : 'primary'}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>{test.sample_size}</TableCell>
                  <TableCell>{test.p_value?.toFixed(4) || 'N/A'}</TableCell>
                  <TableCell>{test.winner || 'TBD'}</TableCell>
                  <TableCell>
                    <Button size="small">View Results</Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </>
      )}

      {tabValue === 2 && (
        <>
          <Button 
            variant="contained" 
            startIcon={<CloudUpload />}
            onClick={() => setUploadDialogOpen(true)}
            sx={{ mb: 3 }}
          >
            Upload New Model
          </Button>
          <Alert severity="info">
            Upload your trained models to share with the community. All models are reviewed before publication.
          </Alert>
        </>
      )}
    </Container>
  );
};

export default MLMarketplace;
